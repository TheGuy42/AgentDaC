from datetime import datetime
from art.utils import output_dirs
from src.configs.base_config import BaseConfig


class PathConfig(BaseConfig, frozen=False):
    base_model: str
    project_name: str
    run_name: str = ""
    art_path: str = ""

    def model_post_init(self, context) -> None:
        """Generate run name if not provided"""
        if self.art_path == "":
            self.art_path = output_dirs.get_default_art_path()

        if self.run_name == "":
            self.run_name = self._generate_run_name(self.base_model)

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
