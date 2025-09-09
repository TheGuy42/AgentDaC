from src.configs.base_config import BaseConfig


class PromptConfig(BaseConfig):
    system_root: str = ""
    system_inter: str = ""
    system_leaf: str = ""
    tasks_depleted: str = ""
