from src.configs.models.vllm._registry import add_config, available_configs, CONFIGS
from src.configs.vllm_config import VllmConfig

__all__ = [
    "VllmConfig",
    "add_config",
    "available_configs",
    "CONFIGS",
]


# import modules to register relevant configs
import src.configs.models.vllm.qwen_models  # noqa: F401
import src.configs.models.vllm.llama_models  # noqa: F401
