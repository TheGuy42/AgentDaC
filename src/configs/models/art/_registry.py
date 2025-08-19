from __future__ import annotations

from art.dev import (
    InternalModelConfig,
    InitArgs,
    EngineArgs,
    PeftArgs,
    TrainerArgs,
    TorchtuneArgs,
    OpenAIServerConfig,
    ServerArgs,
)

from src.utils.logging import create_logger
from src.configs.art_config import ArtConfig


logger = create_logger(__name__)


CONFIGS: dict[str, ArtConfig] = {}


def add_config(
    model_name: str,
    config_id: str = "",
    init_args: InitArgs | None = None,
    engine_args: EngineArgs | None = None,
    peft_args: PeftArgs | None = None,
    trainer_args: TrainerArgs | None = None,
    torchtune_args: TorchtuneArgs | None = None,
    openai_config: OpenAIServerConfig | None = None,
    allow_override: bool = False,
    **kwargs,
):
    """
    Add a configuration to the global CONFIGS dictionary.
    """

    inter_args = {
        "init_args": init_args,
        "engine_args": engine_args,
        "peft_args": peft_args,
        "trainer_args": trainer_args,
        "torchtune_args": torchtune_args,
    }

    inter_args = {k: v for k, v in inter_args.items() if v is not None}
    inter_args.update(kwargs)

    config = ArtConfig(
        id=config_id,
        base_model=model_name,
        internal_config=InternalModelConfig(**inter_args),
        openai_config=openai_config,
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
