from __future__ import annotations

import os
import pathlib

# NOTE: Some ART library black-magic -- must precede `import art`.
os.environ["IMPORT_UNSLOTH"] = "1"
os.environ["IMPORT_PEFT"] = "1"

import asyncio
from typing import Any


from src.backends.art.config import ArtConfig
from src.backends.art.loaders import load_art_model
from src.backends.art.paths import PathConfig
from src.backends.art.trainer import ArtTrainer
from src.configs import DataConfig, RolloutConfig, PromptConfig, DecompConfig
from src.running.stage import RolloutStage
from src.utils.io import load_object
from src.utils.logging import create_logger
from src.utils.trajectory_writer import TrajectoryWriter
from experiments._framework.backends.backend import Backend


logger = create_logger(__name__)


class ArtBackend(Backend):
    @property
    def name(self) -> str:
        return "art"

    def load_configs(self, config_dir: pathlib.Path) -> dict[str, Any]:
        return {
            "data_config": DataConfig.load_from_path(config_dir / "data_config.json", do_raise=True),
            "prompt_config": PromptConfig.load_from_path(config_dir / "prompt_config.json", do_raise=True),
            "decomp_config": DecompConfig.load_from_path(config_dir / "decomp_config.json", do_raise=True),
            "rollout_config": RolloutConfig.load_from_path(config_dir / "art_rollout_config.json", do_raise=True),
            "art_config": ArtConfig.load_from_path(config_dir / "art_config.json", do_raise=True),
            "extra_config": load_object(config_dir / "extra_config.json", do_raise=False) or {},
        }

    def launch(self, configs: dict[str, Any]) -> None:
        asyncio.run(self._launch(configs))

    async def _launch(self, configs: dict[str, Any]) -> None:
        args = self.args.args
        art_config: ArtConfig = configs["art_config"]
        data_config: DataConfig = configs["data_config"]

        if args.test_run:
            self._patch_test_run(art_config, data_config)

        exp_name = args.run or self.default_run_name(art_config.model.base_model)
        logger.info(f"Experiment name: {exp_name}")

        writer_config = self.traj_writer_config(exp_name)
        writer = TrajectoryWriter(writer_config["dir"], enabled=writer_config["enabled"])

        paths = PathConfig(
            base_model=art_config.model.base_model,
            project_name=args.project,
            run_name=exp_name,
        )

        model = await load_art_model(paths, art_config.model, seed=args.seed)

        # Prepare the dataset and the splits, sa
        dataset = self.args.dataset_cls(data_config.model_dump())
        splits_names = dataset.splits.keys()

        split_sizes = {
            RolloutStage.TRAIN: data_config.train_size,
            RolloutStage.VAL: data_config.val_size,
            RolloutStage.TEST: data_config.test_size,
        }

        data_dict = {stage: dataset.prepare_split(stage, split_sizes[stage], data_config.seed).to_list() for stage in splits_names}

        trainer = ArtTrainer(
            model=model,
            task=self.args.task_cls(configs),
            config=art_config,
            rollout_config=configs["rollout_config"],
            agent_name=self.args.agent_name,
            prompt_config=configs["prompt_config"],
            decomp_config=configs["decomp_config"],
            extra_config=configs["extra_config"],
            writer=writer,
        )

        if trainer.wandb_run is not None:
            for root in ("src", "experiments"):
                trainer.wandb_run.log_code(root=root, name=root)

        try:
            logger.info("Starting training...")

            await trainer.train(
                train_dataset=data_dict[RolloutStage.TRAIN],
                val_dataset=data_dict.get(RolloutStage.VAL),
            )

            for stage in (RolloutStage.VAL, RolloutStage.TEST):
                if stage in data_dict:
                    logger.info(f"Starting {stage.value}-set evaluation...")
                    groups = await trainer.rollout(data_dict[stage], group_size=1, step=0, stage=stage)
                    await trainer.model.log(groups, split=stage.value)

        finally:
            await trainer.close()

    def _patch_test_run(self, art_config: ArtConfig, data_config: DataConfig) -> None:
        logger.info("Test run: overriding ART config for a quick run.")
        art_config.train.epochs = 1
        art_config.train.num_groups = 2
        art_config.train.group_size = 2
        art_config.train.val_log_steps = 1
        art_config.train.delete_checkpoints = False
        data_config.train_size = 20
        data_config.val_size = 10
