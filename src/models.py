from datetime import datetime
import logging
from pydantic import BaseModel

import art
from art.utils import output_dirs
from art.local import LocalBackend
from art.dev import OpenAIServerConfig, InternalModelConfig

from src.configs import art_model_config
from src.configs import vllm_model_config


logger = logging.getLogger(__name__)


class PathConfig(BaseModel, frozen=False):
    model_name: str
    project_name: str
    run_name: str = ""
    art_path: str = ""

    def model_post_init(self, context) -> None:
        """Generate run name if not provided"""
        if not self.art_path:
            self.art_path = output_dirs.get_default_art_path()

        if not self.run_name:
            self.run_name = self._generate_run_name(self.model_name)

    @property
    def model_output_dir(self) -> str:
        """
        Get the output directory for the model.
        """
        return output_dirs.get_output_dir_from_model_properties(
            project=self.project_name,
            name=self.model_name,
            art_path=self.art_path,
        )

    @property
    def trajectories_dir(self) -> str:
        return output_dirs.get_trajectories_dir(
            model_output_dir=self.model_output_dir,
        )

    def _generate_run_name(self, model_name: str) -> str:
        """
        Generate a run name based on the model name and current date.
        """
        model_name = self.model_name.split("/")[-1]
        date_str = datetime.now().strftime("%m_%d_%H_%M")
        return f"{model_name}_{date_str}"

    def get_step_checkpoint_dir(self, step: int) -> str:
        """
        Get the checkpoint directory for a specific step.
        """
        return output_dirs.get_step_checkpoint_dir(model_output_dir=self.model_output_dir, step=step)


async def load_art_model(
    path_config: PathConfig,
    internal_config: InternalModelConfig | None = None,
    openai_config: OpenAIServerConfig | None = None,
    print_full: bool = False,
) -> art.TrainableModel:
    if internal_config is None:
        if path_config.model_name not in art_model_config.CONFIGS:
            raise ValueError(
                f"No configuration found for model: {path_config.model_name}. "
                f"Available configs: {art_model_config.available_configs()}"
            )

        logger.info(f"Loading default `internal_config` for {path_config.model_name}...")
        art_config = art_model_config.CONFIGS[path_config.model_name]
        internal_config = art_config.internal_config

    art_config = art_model_config.ArtConfig(
        model_name=path_config.model_name,
        internal_config=internal_config,
    )

    if not print_full:
        print("Model configuration:")
        print(art_config.model_dump_json(indent=4))

    art_config = art_config.to_full(output_dir=path_config.model_output_dir)

    if print_full:
        print("Full model configuration:")
        print(art_config.model_dump_json(indent=4))

    model = art.TrainableModel(
        name=path_config.run_name,
        project=path_config.project_name,
        base_model=path_config.model_name,
        _internal_config=internal_config,
    )

    backend = LocalBackend(path=path_config.art_path)
    await model.register(backend, _openai_client_config=openai_config)
    return model


def load_vllm_model(
    model_name: str,
    port: int = 8200,
    openai_config: OpenAIServerConfig | None = None,
    print_full: bool = False,
) -> list[str]:
    if openai_config is None:
        if model_name not in vllm_model_config.CONFIGS:
            raise ValueError(
                f"No configuration found for model: {model_name}. "
                f"Available configs: {vllm_model_config.available_configs()}"
            )

        logger.info(f"Loading default `openai_config` for {model_name}...")
        vllm_config = vllm_model_config.CONFIGS[model_name]
        openai_config = vllm_config.openai_config

    vllm_config = vllm_model_config.VllmConfig(
        model_name=model_name,
        openai_config=openai_config,
    )

    vllm_config.openai_config.setdefault("server_args", {})["port"] = port

    if not print_full:
        print("Model configuration:")
        print(vllm_config.model_dump_json(indent=4))

    vllm_config = vllm_config.to_full()

    if print_full:
        print("Full model configuration:")
        print(vllm_config.model_dump_json(indent=4))

    engine_args = vllm_config.openai_config.get("engine_args", {})
    server_args = vllm_config.openai_config.get("server_args", {})

    full_args = [
        *[
            f"--{key.replace('_', '-')}{f'={item}' if item is not True else ''}"
            for args in [engine_args, server_args]
            for key, value in args.items()
            for item in (value if isinstance(value, list) else [value])
            if item is not None
        ],
    ]

    return full_args
