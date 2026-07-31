from __future__ import annotations

import pathlib
from typing import Any

from omegaconf import OmegaConf
from transformers import AutoConfig

import verl
from verl.trainer.main_ppo import run_ppo
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils.config import validate_config
from verl.utils.device import auto_set_device

from src.backends.verl.trainer import CustomTaskRunner
from src.backends.verl.template import resolve_chat_template
from src.configs import DataConfig, RolloutConfig, PromptConfig, DecompConfig
from src.utils.io import load_object
from src.utils.logging import create_logger
from experiments._framework.backends.backend import Backend


logger = create_logger(__name__)


# Repo root, so the Ray workers can import experiment `_target_` FQDNs
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


class VerlBackend(Backend):
    @property
    def name(self) -> str:
        return "verl"

    def load_configs(self, config_dir: pathlib.Path) -> dict[str, Any]:
        return {
            "data_config": DataConfig.load_from_path(config_dir / "data_config.json", do_raise=True),
            "prompt_config": PromptConfig.load_from_path(config_dir / "prompt_config.json", do_raise=True),
            "decomp_config": DecompConfig.load_from_path(config_dir / "decomp_config.json", do_raise=True),
            "rollout_config": RolloutConfig.load_from_path(config_dir / "verl_rollout_config.json", do_raise=True),
            "verl_config": load_object(config_dir / "verl_config.json", do_raise=True),  # OVERRIDES onto verl defaults
            "extra_config": load_object(config_dir / "extra_config.json", do_raise=False) or {},
        }

    def launch(self, configs: dict[str, Any]) -> None:
        args = self.args.args

        base_model = str(configs["verl_config"]["actor_rollout_ref"]["model"]["path"])
        exp_name = args.run or self.default_run_name(base_model)
        logger.info(f"Experiment name: {exp_name}")

        # verl constructs the real TrajectoryWriter inside VerlLoop, in the Ray worker.
        configs["traj_writer"] = self.traj_writer_config(exp_name)

        if args.test_run:
            data_config: DataConfig = configs["data_config"]
            data_config.train_size = 20
            data_config.val_size = 10

        config = self._build_verl_config(configs, exp_name)
        logger.info("Starting training (main_ppo_sync)...")
        self._launch(config)

    def _verl_default_config(self) -> Any:
        """The complete flattened verl `ppo_trainer` default config (our merge base).

        `main_ppo_sync` reads the whole schema, so a partial `verl_config.json` is not enough.
        We load verl's shipped flattened default and merge the experiment overrides onto it.
        (Equivalent canonical form: `hydra.compose(config_name="ppo_trainer")`.)
        """
        generated = pathlib.Path(verl.__file__).parent / "trainer" / "config" / "_generated_ppo_trainer.yaml"
        return OmegaConf.load(generated)

    def _build_verl_config(self, configs: dict[str, Any], exp_name: str) -> Any:
        args = self.args.args
        data_config: DataConfig = configs["data_config"]

        overrides = dict(configs.pop("verl_config"))
        omega_conf = OmegaConf.merge(self._verl_default_config(), OmegaConf.create(overrides))
        OmegaConf.set_struct(omega_conf, False)

        # Naming and seed. Resume is set natively in verl_config.json, via
        # `trainer.resume_mode` / `trainer.resume_from_path`.
        omega_conf.trainer.project_name = args.project
        omega_conf.trainer.experiment_name = exp_name
        omega_conf.data.seed = args.seed

        # TransferQueue is required by main_ppo_sync (default enable=False).
        omega_conf.transfer_queue.enable = True

        # Dataset: verl builds it in-worker via `data.custom_cls` (no parquet on disk).
        # `train_files`/`val_files` are split markers the dataset class branches on;
        # `custom_dataset` is the whole `data_config.json` (sizes, dataset seed, load params),
        # which is what the dataset reads as `TaskDataset.params`.
        dataset_cls = self.args.dataset_cls
        omega_conf.data.custom_cls = {"path": "pkg://src.backends.verl.dataset", "name": "VerlDataset"}
        omega_conf.data.train_files = "train"
        omega_conf.data.val_files = "val"
        omega_conf.data.custom_dataset = {
            **data_config.model_dump(),
            "task_dataset": f"{dataset_cls.__module__}.{dataset_cls.__qualname__}",
            "data_source": args.project,
        }

        # Agent-loop registration: a committed file, since `VerlLoop` is the `_target_` for every experiment.
        omega_conf.actor_rollout_ref.rollout.agent.agent_loop_config_path = str(REPO_ROOT / "src/backends/verl/loop.yaml")
        omega_conf.actor_rollout_ref.rollout.agent.default_agent_loop = "agentdac"

        # On-policy + agent-loop engine invariants.
        omega_conf.actor_rollout_ref.rollout.mode = "async"  # AsyncLLM engine (NOT an off-policy switch)
        if int(omega_conf.actor_rollout_ref.rollout.nnodes) != 0:
            raise ValueError(
                f"rollout.nnodes must be 0 (colocated rollout) for on-policy training; got {omega_conf.actor_rollout_ref.rollout.nnodes}."
            )

        # Embed the custom configs so the per-sample VerlLoop can rebuild them.
        omega_conf.custom_configs = {k: (v.model_dump() if hasattr(v, "model_dump") else v) for k, v in configs.items()}

        # Patch the rollout/model lengths.
        self._patch_lengths(omega_conf)

        if args.test_run:
            self._patch_test_run(omega_conf)

        # Make the repo importable inside Ray workers (for the agent-loop `_target_` FQDNs).
        OmegaConf.update(omega_conf, "ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH", str(REPO_ROOT), force_add=True)

        # Verify (and patch, if possible) the chat template before training starts.
        self._patch_chat_template(omega_conf)

        # Inject the agent kind into the verl config so the agent-loop can rebuild it.
        OmegaConf.update(omega_conf, "custom_configs.agent", self.args.agent_name, force_add=True)

        # The experiment's RolloutTask travels as an FQDN, not a class reference: it has to cross a
        # Ray/hydra boundary, and the generic VerlLoop resolves it per rollout. Same mechanism as
        # `custom_dataset.task_dataset` above.
        task_cls = self.args.task_cls
        OmegaConf.update(omega_conf, "custom_configs.task", f"{task_cls.__module__}.{task_cls.__qualname__}", force_add=True)

        return omega_conf

    def _patch_chat_template(self, omega_conf: Any) -> None:
        """Load and verify (and patch, if possible) the chat template before training starts.

        `convert_trajectory` reconstructs the trajectory from per-turn prompt tokens and
        requires a prefix-preserving chat template. Resolve it here -- against the *effective*
        template verl will use -- and inject any patched template via `custom_chat_template`
        so `HFModelConfig` applies it to the tokenizer the rollout shares. Raises if unsafe.
        """
        model_conf = omega_conf.actor_rollout_ref.model
        manual_template = model_conf.get("custom_chat_template", None)

        # check if manual_template is a path to a file, if so, read the file and set manual_template to its contents
        if manual_template is not None and isinstance(manual_template, str) and pathlib.Path(manual_template).is_file():
            logger.info(f"Loading manual chat template from {manual_template} (encoding='utf-8').")
            manual_template = pathlib.Path(manual_template).read_text(encoding="utf-8")
            model_conf.custom_chat_template = manual_template

        resolved = resolve_chat_template(
            model_path=model_conf.path,
            manual_template=model_conf.get("custom_chat_template", None),
            trust_remote_code=bool(model_conf.get("trust_remote_code", False)),
        )

        if resolved is not None:
            model_conf.custom_chat_template = resolved

    def _patch_lengths(self, config: Any) -> None:
        """Patch sizes of rollout/model lengths for multi-turn trajectory."""

        # NOTE DOCS:
        # config.data.max_prompt_length:
        #   Maximum initial prompt length: tokens before the first assistant-generated token.
        #   In multi-turn RL, this should cover system + user prompt + chat template + tool schemas
        #   + generation prompt, but not later assistant/tool turns.

        # config.data.max_response_length:
        #   Maximum cumulative trajectory suffix after the initial prompt.
        #   In multi-turn RL, this includes all generated assistant tokens, tool-call syntax,
        #   tool observations inserted into the conversation, later user/tool messages, and final answer.

        # config.actor_rollout_ref.rollout.prompt_length:
        #   Rollout-side prompt budget. In the default config, it is derived from data.max_prompt_length.

        # config.actor_rollout_ref.rollout.response_length:
        #   Rollout-side cumulative response/trajectory budget. In the default config, it is derived
        #   from data.max_response_length.

        # config.actor_rollout_ref.rollout.max_model_len:
        #   Rollout engine context-window limit for prompt + output at any generation step.
        #   It must be <= the model max_position_embeddings as VERL interprets it.

        prompt_length = int(config.data.max_prompt_length)
        response_length = int(config.data.max_response_length)
        model_length = prompt_length + response_length

        hf_config = AutoConfig.from_pretrained(config.actor_rollout_ref.model.path, trust_remote_code=True)
        hf_model_length: int | None = getattr(hf_config, "max_position_embeddings", getattr(hf_config, "model_max_length", None))

        if hf_model_length and model_length > hf_model_length:
            logger.warning(f"Computed model length exceeds max_position_embeddings: {model_length} > {hf_model_length}. ")
            logger.warning("Decreasing response_length to fit within the model's max_position_embeddings.")
            response_length = hf_model_length - prompt_length
            model_length = hf_model_length

        config.data.max_prompt_length = prompt_length
        config.data.max_response_length = response_length
        config.actor_rollout_ref.rollout.prompt_length = prompt_length
        config.actor_rollout_ref.rollout.response_length = response_length
        config.actor_rollout_ref.rollout.max_model_len = model_length

        logger.info("Patched Lengths:")
        logger.info(f"  prompt_length: {prompt_length}")
        logger.info(f"  response_length: {response_length}")
        logger.info(f"  max_model_len: {model_length}")

    def _patch_test_run(self, config: Any) -> None:
        logger.info("Test run: overriding verl config for a quick run.")
        config.trainer.logger = ["console"]
        config.trainer.nnodes = 1
        config.trainer.n_gpus_per_node = 1
        config.trainer.test_freq = 1
        config.trainer.save_freq = -1
        config.trainer.total_epochs = 1
        config.trainer.total_training_steps = 2
        config.data.train_batch_size = 6
        config.actor_rollout_ref.rollout.n = 2
        config.actor_rollout_ref.actor.ppo_mini_batch_size = 2
        config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu = 2
        config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu = 2

    def _launch(self, config: Any) -> None:
        """Replicate `main_ppo_sync.main` (we bypass its @hydra.main entrypoint)."""

        auto_set_device(config)
        config.transfer_queue.enable = True
        validate_config(
            config=config,
            use_reference_policy=need_reference_policy(config),
            use_critic=need_critic(config),
        )
        run_ppo(config, task_runner_class=CustomTaskRunner)
