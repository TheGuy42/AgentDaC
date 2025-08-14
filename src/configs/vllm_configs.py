from __future__ import annotations
from pydantic import BaseModel, Field, model_validator
from pathlib import Path
from art.dev import ServerArgs, EngineArgs, OpenAIServerConfig, get_openai_server_config
from src.utils.io import save_base_model


# TODO: test following settings:
# There are also 8-bit quantized versions of models, test them.


class VllmConfig(BaseModel, frozen=False, extra="allow"):
    """
    Configuration for a served vLLM model.
    """

    id: str = ""  # NOTE: not supported yet
    base_model: str
    openai_config: OpenAIServerConfig = Field(default_factory=OpenAIServerConfig)

    @model_validator(mode="after")
    def validate_identifier(self) -> VllmConfig:
        """
        Validate that the identifier is set to the base model name if not provided.
        """
        if not self.id:
            self.id = self.base_model
        return self

    def initialize(self, port: int) -> VllmConfig:
        self.openai_config = get_openai_server_config(
            model_name=self.base_model,
            base_model=self.base_model,
            log_file="",
            lora_path=None,
            config=self.openai_config,
        )
        self.openai_config["server_args"]["port"] = port  # type: ignore
        self.openai_config["engine_args"]["enable_lora"] = True  # type: ignore
        self.openai_config["engine_args"].setdefault("seed", 0)  # type: ignore
        return self

    def save(self, dir_name: str, file_name: str = "vllm_config.json") -> None:
        save_base_model(self, Path(dir_name) / file_name)


CONFIGS: dict[str, VllmConfig] = {}


def add_config(
    model_name: str,
    config_id: str = "",
    server_args: ServerArgs | None = None,
    engine_args: EngineArgs | None = None,
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
        raise ValueError(f"Configuration for '{config.id}' already exists.")

    CONFIGS[config.id] = config


def available_configs() -> list[str]:
    """
    Returns a list of available model configurations.
    """
    return sorted(list(CONFIGS.keys()))


################################################################
##################### Model Configurations #####################
################################################################

add_config(
    "unsloth/Qwen2-7B",
    engine_args=EngineArgs(
        max_model_len=4096 * 2,
        max_num_batched_tokens=4096 * 4 * 2,
        max_seq_len_to_capture=4096 * 2,
        gpu_memory_utilization=0.8,
    ),
)

add_config(
    "unsloth/Qwen2.5-14B-Instruct",
    engine_args=EngineArgs(
        max_model_len=4096 * 2,
        max_num_batched_tokens=4096 * 4 * 2,
        max_seq_len_to_capture=4096 * 2,
        gpu_memory_utilization=0.8,
    ),
)

add_config(
    "unsloth/Qwen2.5-14B-Instruct-bnb-4bit",
    engine_args=EngineArgs(
        max_num_seqs=128,
        gpu_memory_utilization=0.9,
        load_format="bitsandbytes",
    ),
)

add_config(
    "unsloth/Qwen2.5-32B-Instruct",
    engine_args=EngineArgs(
        max_num_seqs=128,
        max_model_len=10000,
        gpu_memory_utilization=0.85,
    ),
)

add_config(
    "unsloth/Qwen3-14B",
    engine_args=EngineArgs(
        max_model_len=1024*16,
        gpu_memory_utilization=0.9,
    ),
)

add_config(
    "unsloth/Llama-4-Scout-17B-16E-Instruct",
    engine_args=EngineArgs(
        max_num_seqs=128,
        gpu_memory_utilization=0.85,
    ),
)

add_config(
    "unsloth/Llama-4-Scout-17B-16E-Instruct-unsloth-dynamic-bnb-4bit",
    engine_args=EngineArgs(
        max_num_seqs=128,
        gpu_memory_utilization=0.85,
        load_format="bitsandbytes",
    ),
)


