from __future__ import annotations
import asyncio
import pathlib
from typing import Any

from src.backends.vllm.config import VllmConfig
from src.backends.vllm.runner import VllmRunner
from src.configs import DataConfig, DecompConfig, PromptConfig, RolloutConfig
from src.running.stage import RolloutStage
from src.utils.io import load_object
from src.utils.logging import create_logger
from src.utils.trajectory_writer import TrajectoryWriter
from experiments._framework.backends.backend import Backend


logger = create_logger(__name__)


class VllmBackend(Backend):
    """Inference-only backend over a live `vllm serve`. Trains nothing."""

    @property
    def name(self) -> str:
        return "vllm"

    def load_configs(self, config_dir: pathlib.Path) -> dict[str, Any]:
        return {
            "data_config": DataConfig.load_from_path(config_dir / "data_config.json", do_raise=True),
            "prompt_config": PromptConfig.load_from_path(config_dir / "prompt_config.json", do_raise=True),
            "decomp_config": DecompConfig.load_from_path(config_dir / "decomp_config.json", do_raise=True),
            "rollout_config": RolloutConfig.load_from_path(config_dir / "vllm_rollout_config.json", do_raise=True),
            "vllm_config": VllmConfig.load_from_path(config_dir / "vllm_config.json", do_raise=True),
            "extra_config": load_object(config_dir / "extra_config.json", do_raise=False) or {},
        }

    def launch(self, configs: dict[str, Any]) -> None:
        asyncio.run(self._launch(configs))

    async def _launch(self, configs: dict[str, Any]) -> None:
        args = self.args.args
        vllm_config: VllmConfig = configs["vllm_config"]
        data_config: DataConfig = configs["data_config"]
        inference = vllm_config.inference

        if args.test_run:
            data_config.train_size = 10
            data_config.val_size = 10
            data_config.test_size = 10
            inference.group_size = 1

        logger.info(f"Serving model: {vllm_config.server.model_name} at {vllm_config.server.base_url}")

        exp_name = args.run or self.default_run_name(vllm_config.server.model_name)
        writer_config = self.traj_writer_config(exp_name)

        runner = VllmRunner(
            config=vllm_config,
            task=self.args.task_cls(configs),
            agent_name=self.args.agent_name,
            prompt_config=configs["prompt_config"],
            decomp_config=configs["decomp_config"],
            rollout_config=configs["rollout_config"],
            extra_config=configs["extra_config"],
            writer=TrajectoryWriter(writer_config["dir"], enabled=writer_config["enabled"]),
            run_name=exp_name,
        )

        dataset = self.args.dataset_cls(data_config.model_dump())
        sizes = {
            RolloutStage.TRAIN: data_config.train_size,
            RolloutStage.VAL: data_config.val_size,
            RolloutStage.TEST: data_config.test_size,
        }

        datasets = {}
        for stage in inference.splits:
            if stage not in dataset.splits:
                logger.warning(f"Dataset provides no {stage.value!r} split; skipping.")
                continue
            datasets[stage] = dataset.prepare_split(stage, sizes[stage], data_config.seed).to_list()

        try:
            await runner.evaluate(datasets, step=0)
        finally:
            runner.close()
