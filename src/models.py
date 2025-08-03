from datetime import datetime
from pydantic import BaseModel
from pathlib import Path

import art
from art.utils import output_dirs
from art.local import LocalBackend

from src.configs import art_configs
from src.configs import vllm_configs
from src.utils.logging import create_logger
from src.utils.io import save_base_model


logger = create_logger(__name__)


class PathConfig(BaseModel, frozen=False):
    base_model: str
    project_name: str
    run_name: str = ""
    art_path: str = ""

    def model_post_init(self, context) -> None:
        """Generate run name if not provided"""
        if not self.art_path:
            self.art_path = output_dirs.get_default_art_path()

        if not self.run_name:
            self.run_name = self._generate_run_name(self.base_model)

    def save(self, dir_name: str, file_name: str = "path_config.json") -> None:
        """
        Save the path configuration to a JSON file.
        """
        save_base_model(self, Path(dir_name) / file_name)

    @property
    def model_output_dir(self) -> str:
        """
        Get the output directory for the model.
        """
        return output_dirs.get_output_dir_from_model_properties(
            project=self.project_name,
            name=self.run_name,
            art_path=self.art_path,
        )

    @property
    def trajectories_dir(self) -> str:
        return output_dirs.get_trajectories_dir(
            model_output_dir=self.model_output_dir,
        )

    def _generate_run_name(self, base_model: str) -> str:
        """
        Generate a run name based on the model name and current date.
        """
        base_model = self.base_model.split("/")[-1]
        date_str = datetime.now().strftime("%m_%d_%H_%M")
        return f"{base_model}_{date_str}"

    def get_step_checkpoint_dir(self, step: int) -> str:
        """
        Get the checkpoint directory for a specific step.
        """
        return output_dirs.get_step_checkpoint_dir(model_output_dir=self.model_output_dir, step=step)

async def load_art_model(
    path_config: PathConfig,
    art_config: art_configs.ArtConfig | None = None,
    print_full: bool = False,
) -> art.TrainableModel:
    if art_config is None:
        if path_config.base_model not in art_configs.CONFIGS:
            raise ValueError(
                f"No configuration found for model: {path_config.base_model}. "
                f"Available configs: {art_configs.available_configs()}"
            )

        logger.info(f"Loading default config for {path_config.base_model}...")
        art_config = art_configs.CONFIGS[path_config.base_model]

    if path_config.base_model != art_config.base_model:
        raise ValueError(f"Model name mismatch: {path_config.base_model} != {art_config.base_model}.")

    if not print_full:
        print("Art Config:")
        print(art_config.model_dump_json(indent=4))

    art_config = art_config.initialize(output_dir=path_config.model_output_dir)

    if print_full:
        print("Full Art Config:")
        print(art_config.model_dump_json(indent=4))

    model = art.TrainableModel(
        name=path_config.run_name,
        project=path_config.project_name,
        base_model=path_config.base_model,
        _internal_config=art_config.internal_config,
    )

    backend = LocalBackend(path=path_config.art_path)
    await model.register(backend, _openai_client_config=art_config.openai_config)
    return model


def load_vllm_model(
    model_name: str,
    port: int = 8200,
    vllm_config: vllm_configs.VllmConfig | None = None,
    print_full: bool = False,
) -> list[str]:
    if vllm_config is None:
        if model_name not in vllm_configs.CONFIGS:
            raise ValueError(
                f"No configuration found for model: {model_name}. Available configs: {vllm_configs.available_configs()}"
            )

        logger.info(f"Loading default config for {model_name}...")
        vllm_config = vllm_configs.CONFIGS[model_name]

    if vllm_config.base_model != model_name:
        raise ValueError(f"Model name mismatch: {vllm_config.base_model} != {model_name}.")

    if not print_full:
        print("vLLM Config:")
        print(vllm_config.model_dump_json(indent=4))

    vllm_config = vllm_config.initialize(port)

    if print_full:
        print("Full vLLM Config:")
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
