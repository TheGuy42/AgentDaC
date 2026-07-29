from __future__ import annotations

import argparse
import atexit
import logging
import os
import pathlib
import random
import sys
import tempfile
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

import torch
from omegaconf import OmegaConf
from transformers import AutoConfig

from src.configs import DecompConfig, PromptConfig, RolloutConfig, TrainingConfig
from src.utils.env import prepare_environment, set_seed
from src.utils.io import load_object
from src.utils.logging import create_logger, setup_logging
from src.utils.chat_template import resolve_chat_template
from src.running.dataset import TaskDataset

logger = create_logger(__name__)

# Repo root, so the Ray workers can import experiment `_target_` FQDNs
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

# TODO: clean up this overall file, we need to make it more clean
# and better structured / more maintainable
# but also overall code readability is important, and things shouldnt be scattered around too much


class ExperimentRunner(ABC):
    def __init__(self) -> None:
        self._parser_args = None

    def args(self) -> argparse.Namespace:
        if self._parser_args is None:
            raise ValueError("Arguments have not been parsed yet. Call _parse_args() first.")
        return self._parser_args

    @abstractmethod
    def task_name(self) -> str:
        """Short task identifier, e.g. 'math'. Roots the config dir and project name."""

    @abstractmethod
    def supported_agents(self) -> list[str]:
        """Agent kinds this task supports."""

    def default_project_name(self) -> str:
        """Override to specify default project name."""
        return f"{self.task_name()}_{self.args().agent}"

    def default_config_dir(self) -> str:
        """Override to specify the experiment's config directory (holds the JSON configs)."""
        return f"experiments/{self.task_name()}/configs/{self.args().agent}"

    @abstractmethod
    def dataset_class(self) -> type[TaskDataset]:
        """Return the experiment's `TaskDataset` subclass."""

    @abstractmethod
    def trainer_class(self) -> type:
        """Return the experiment's `VerlLoop` subclass — the verl agent-loop `_target_`.

        Used to generate the agent-loop registration at runtime (see `_write_agent_loop_yaml`),
        replacing a committed `agent_loop.yaml`."""

    def dataset_args(self) -> dict[str, Any]:
        """Override to provide experiment-specific dataset params.

        Embedded under `config.data.custom_dataset` and read by the dataset class's
        `load_split` (e.g. `{"min_level": ..., "max_level": ...}`)."""
        return {}

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Override to add custom command line arguments."""

    def default_run_name(self, base_model: str) -> str:
        """Default run name to use."""
        base_model = base_model.split("/")[-1]
        date_str = datetime.now().strftime("%m_%d_%H_%M")
        return f"{base_model}_{date_str}"

    def _parse_args(self) -> argparse.Namespace:
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        parser.add_argument(
            "--agent",
            type=str,
            required=True,
            help="The agent kind to run",
            choices=self.supported_agents(),
        )

        parser.add_argument(
            "--project",
            type=str,
            default=None,
            help="Project name. If not provided, defaults to `default_project_name()`.",
        )

        parser.add_argument(
            "--run",
            type=str,
            default=None,
            help="Experiment run name. If not provided, defaults to `default_run_name()`.",
        )

        parser.add_argument(
            "--resume",
            type=str,
            default=None,
            help="Checkpoint path to resume from.",
        )

        parser.add_argument(
            "--gpus",
            type=int,
            nargs="+",
            default=[0],
            help="GPU IDs to use.",
        )

        parser.add_argument(
            "--config_dir",
            type=str,
            default=None,
            help="Config directory. If not provided, defaults to `default_config_dir()`.",
        )

        parser.add_argument(
            "--traj_dir",
            type=str,
            default=None,
            help=(
                "Base directory for logging full rollout trajectories to disk "
                "(one JSON file per rollout, grouped by training step). Disabled when not provided."
            ),
        )

        parser.add_argument(
            "--seed",
            type=int,
            default=random.randint(0, 1000000),
            help="Random seed.",
        )

        parser.add_argument(
            "--silent",
            action="store_true",
            help="Disable verbose outputs.",
        )

        parser.add_argument(
            "--test_run",
            action="store_true",
            help="Quick minimal run for debugging.",
        )

        self.add_arguments(parser)
        self._parser_args = parser.parse_args()
        args = self.args()

        if args.project is None:
            args.project = self.default_project_name()

        if args.config_dir is None:
            args.config_dir = self.default_config_dir()

        if not all(0 <= gpu < torch.cuda.device_count() for gpu in args.gpus):
            raise ValueError(f"Invalid GPU IDs {args.gpus}. Available: {list(range(torch.cuda.device_count()))}")

        print("\nParsed arguments:")
        for arg, value in vars(args).items():
            print(f"  {arg}: {value}")
        print()
        return args

    def _load_configs(self, dir: str | pathlib.Path) -> dict[str, Any]:
        dir = pathlib.Path(dir)
        return {
            "train_config": TrainingConfig.load_from_path(dir / "train_config.json", do_raise=True),
            "prompt_config": PromptConfig.load_from_path(dir / "prompt_config.json", do_raise=True),
            "decomp_config": DecompConfig.load_from_path(dir / "decomp_config.json", do_raise=True),
            "rollout_config": RolloutConfig.load_from_path(dir / "rollout_config.json", do_raise=True),
            "verl_config": load_object(dir / "verl_config.json", do_raise=True),  # OVERRIDES onto verl defaults
            "extra_config": load_object(dir / "extra_config.json", do_raise=False) or {},
        }

    def _verl_default_config(self) -> Any:
        """The complete flattened verl `ppo_trainer` default config (our merge base).

        `main_ppo_sync` reads the whole schema, so a partial `verl_config.json` is not enough.
        We load verl's shipped flattened default and merge the experiment overrides onto it.
        (Equivalent canonical form: `hydra.compose(config_name="ppo_trainer")`.)
        """
        import verl

        generated = pathlib.Path(verl.__file__).parent / "trainer" / "config" / "_generated_ppo_trainer.yaml"
        return OmegaConf.load(generated)

    def _write_agent_loop_yaml(self, name: str, target: str) -> pathlib.Path:
        """Write the verl agent-loop registration to a temp yaml and return its path.

        verl's `AgentLoopWorker` loads this via `OmegaConf.load(agent_loop_config_path)`, so it
        must be a real file; we generate it from `trainer_class()` instead of committing one."""
        entries = OmegaConf.create([{"name": name, "_target_": target}])
        fd, path = tempfile.mkstemp(prefix=f"agent_loop_{name}_", suffix=".yaml")
        os.close(fd)
        OmegaConf.save(entries, path)
        atexit.register(lambda: pathlib.Path(path).unlink(missing_ok=True))
        return pathlib.Path(path)

    def _build_verl_config(self, configs: dict[str, Any], exp_name: str) -> Any:
        args = self.args()
        train_config: TrainingConfig = configs["train_config"]

        overrides = dict(configs.pop("verl_config"))
        omega_conf = OmegaConf.merge(self._verl_default_config(), OmegaConf.create(overrides))
        OmegaConf.set_struct(omega_conf, False)

        # Naming, seed, resume.
        omega_conf.trainer.project_name = args.project
        omega_conf.trainer.experiment_name = exp_name
        omega_conf.data.seed = args.seed
        if args.resume:
            omega_conf.trainer.resume_mode = "resume_path"
            omega_conf.trainer.resume_from_path = args.resume

        # TransferQueue is required by main_ppo_sync (default enable=False).
        omega_conf.transfer_queue.enable = True

        # Dataset: verl builds it in-worker via `data.custom_cls` (no parquet on disk).
        # `train_files`/`val_files` are split markers the dataset class branches on;
        # `custom_dataset` carries the load params (the dataset only sees `config.data`).
        cls = self.dataset_class()
        omega_conf.data.custom_cls = {"path": "pkg://src.backends.verl.dataset", "name": "VerlDataset"}
        omega_conf.data.train_files = "train"
        omega_conf.data.val_files = "val"
        omega_conf.data.custom_dataset = {
            **self.dataset_args(),
            "task_dataset": f"{cls.__module__}.{cls.__qualname__}",
            "train_size": train_config.train_size,
            "val_size": train_config.val_size,
            "seed": args.seed,
            "data_source": args.project,
        }

        # Agent-loop registration: generated at runtime from `trainer_class()` (verl loads it
        # via OmegaConf.load, so it must be a real file).
        trainer_cls = self.trainer_class()
        agent_name = trainer_cls.__name__
        agent_loop_path = self._write_agent_loop_yaml(agent_name, f"{trainer_cls.__module__}.{trainer_cls.__qualname__}")
        omega_conf.actor_rollout_ref.rollout.agent.agent_loop_config_path = str(agent_loop_path)
        omega_conf.actor_rollout_ref.rollout.agent.default_agent_loop = agent_name

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
        OmegaConf.update(omega_conf, "custom_configs.agent", self.args().agent, force_add=True)

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

        # DOCS:
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
        from verl.trainer.main_ppo import run_ppo
        from verl.trainer.ppo.utils import need_critic, need_reference_policy
        from verl.utils.config import validate_config
        from verl.utils.device import auto_set_device
        from src.backends.verl.trainer import CustomTaskRunner

        auto_set_device(config)
        config.transfer_queue.enable = True
        validate_config(
            config=config,
            use_reference_policy=need_reference_policy(config),
            use_critic=need_critic(config),
        )
        run_ppo(config, task_runner_class=CustomTaskRunner)

    def _main(self) -> None:
        args = self.args()
        logger.info(f"Current working directory: {os.getcwd()}")

        set_seed(args.seed)
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, args.gpus))

        configs = self._load_configs(args.config_dir)

        train_config = configs["train_config"]
        if args.test_run:
            train_config.train_size = 20
            train_config.val_size = 10

        model_name = str(configs["verl_config"]["actor_rollout_ref"]["model"]["path"])
        exp_name = args.run or self.default_run_name(model_name)
        logger.info(f"Experiment name: {exp_name}")

        # Inject the trajectory writer config
        configs["traj_writer"] = {
            "dir": (pathlib.Path(args.traj_dir or "trajectories") / args.project / exp_name).as_posix(),
            "enabled": args.traj_dir is not None,
        }

        config = self._build_verl_config(configs, exp_name)
        logger.info("Starting training (main_ppo_sync)...")
        self._launch(config)

    def run(self) -> None:
        try:
            self._parse_args()
            prepare_environment()
            setup_logging(level=logging.WARNING if self.args().silent else logging.INFO)
            self._main()
        except KeyboardInterrupt:
            logger.info("Training interrupted by user.")
            sys.exit(0)
