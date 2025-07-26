from datetime import datetime

import art
from art.utils import output_dirs
from art.dev import InternalModelConfig
from art.local import LocalBackend

from src.configs import art_model_config
from src.configs import vllm_model_config


class DirConfig:
    def __init__(
        self,
        model_name: str,
        project_name: str,
        art_path: str,
    ) -> None:
        self.model_name = model_name
        self.project_name = project_name
        self.art_path = art_path
        self.run_name = self._generate_run_name(model_name)

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
    model_name: str,
    project_name: str,
    backend: LocalBackend,
    config: InternalModelConfig | None = None,
) -> tuple[art.TrainableModel, DirConfig]:
    if config is None:
        config = art_model_config.configs[model_name]

    dir_config = DirConfig(
        model_name=model_name,
        project_name=project_name,
        art_path=backend._path,
    )

    model = art.TrainableModel(
        name=dir_config.run_name,
        project=dir_config.project_name,
        base_model=dir_config.model_name,
        _internal_config=config,
    )

    await model.register(backend)

    return model, dir_config


def load_vllm_model(
    model_name: str,
    server_port: int = 8200,
    gpu_id: int = 1,
):
    
    vllm_config = vllm_model_config.model_configs[model_name]
    vllm_config.run_server(port=server_port, gpu=str(gpu_id))