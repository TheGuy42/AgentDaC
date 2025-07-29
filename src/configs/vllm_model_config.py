from __future__ import annotations
from art.dev import ServerArgs, EngineArgs, OpenAIServerConfig, get_openai_server_config
from pydantic import BaseModel, Field


# TODO: test following settings:
# Does it matter if we import original model or unsloth model?
# Does optimizations differ between original and unsloth models?


class VllmConfig(BaseModel, frozen=False):
    """
    Configuration for a vLLM model.
    """

    model_name: str
    server_args: dict = Field(default_factory=ServerArgs)
    engine_args: dict = Field(default_factory=EngineArgs)

    def to_full(self, lora_path: str | None = None) -> VllmConfig:
        openai_config = OpenAIServerConfig(
            server_args=self.server_args,
            engine_args=self.engine_args,
        )
        openai_config = get_openai_server_config(
            model_name=self.model_name,
            base_model=self.model_name,
            log_file="",
            lora_path=lora_path,
            config=openai_config,
        )
        self.engine_args = openai_config.get("engine_args", self.engine_args)
        self.server_args = openai_config.get("server_args", self.server_args)
        self.engine_args["enable_lora"] = True
        return self


CONFIGS: dict[str, VllmConfig] = {}


def add_config(
    *model_names: str,
    server_args: ServerArgs | None = None,
    engine_args: EngineArgs | None = None,
    **kwargs,
):
    """
    Add a configuration to the global CONFIGS dictionary.
    """
    for model_name in model_names:
        if model_name in CONFIGS:
            raise ValueError(f"Configuration for model '{model_name}' already exists.")

        config = VllmConfig(
            model_name=model_name,
            server_args=server_args or ServerArgs(),
            engine_args=engine_args or EngineArgs(),
            **kwargs,
        )

        CONFIGS[model_name] = config


def available_configs() -> list[str]:
    """
    Returns a list of available model configurations.
    """
    return list(CONFIGS.keys())


################################################################
##################### Model Configurations #####################
################################################################


add_config(
    "unsloth/Qwen2.5-32B-Instruct",
    engine_args=EngineArgs(
        max_num_seqs=128,
        max_model_len=10000,
        gpu_memory_utilization=0.85,
    ),
)

add_config(
    "unsloth/Qwen2.5-14B-Instruct",
    engine_args=EngineArgs(
        max_num_seqs=64,
        max_model_len=10000,
        gpu_memory_utilization=0.79,
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
    "unsloth/Qwen3-14B-bnb-4bit",
    engine_args=EngineArgs(
        max_num_seqs=128,
        gpu_memory_utilization=0.7,
        load_format="bitsandbytes",
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
