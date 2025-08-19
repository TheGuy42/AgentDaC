from pydantic import BaseModel
from pathlib import Path
from src.utils.io import save_base_model

class PromptConfig(BaseModel):
    system_root: str = ""
    system_inter: str = ""
    system_leaf: str = ""
    tasks_depleted: str = ""

    def save(self, dir_name: str, file_name: str = "prompt_config.json") -> None:
        """
        Save the prompt configuration to a JSON file.
        """
        save_base_model(self, Path(dir_name) / file_name)
