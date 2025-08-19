from __future__ import annotations
from src.utils.logging import create_logger
from src.configs.vllm_config import VllmConfig
from art.dev import ServerArgs, EngineArgs, OpenAIServerConfig


logger = create_logger(__name__)


CONFIGS: dict[str, VllmConfig] = {}


def add_config(
    model_name: str,
    config_id: str = "",
    server_args: ServerArgs | None = None,
    engine_args: EngineArgs | None = None,
    allow_override: bool = False,
    **kwargs,
):
    """
    Add a configuration to the global CONFIGS dictionary.
    """

    args = {
        "server_args": server_args,
        "engine_args": engine_args,
    }

    args = {k: v for k, v in args.items() if v is not None}
    args.update(kwargs)

    config = VllmConfig(
        id=config_id or model_name,
        base_model=model_name,
        openai_config=OpenAIServerConfig(**args),
        **kwargs,
    )

    if config.id in CONFIGS:
        if allow_override:
            logger.warning(f"Configuration for '{config.id}' already exists. Overriding with this config.")
        else:
            raise ValueError(f"Configuration for '{config.id}' already exists.")

    CONFIGS[config.id] = config


def available_configs() -> list[str]:
    """
    Returns a list of available model configurations.
    """
    return sorted(list(CONFIGS.keys()))
