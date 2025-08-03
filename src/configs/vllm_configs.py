from __future__ import annotations
from art.dev import ServerArgs, EngineArgs, OpenAIServerConfig, get_openai_server_config
from pydantic import BaseModel, Field


# TODO: test following settings:
# There are also 8-bit quantized versions of models, test them.


class VllmConfig(BaseModel, frozen=False, extra="allow"):
    """
    Configuration for a served vLLM model.
    """

    base_model: str
    openai_config: OpenAIServerConfig = Field(default_factory=OpenAIServerConfig)

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
        self.openai_config["engine_args"].setdefault("multi_step_stream_outputs", False)  # type: ignore
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

    args = {
        "server_args": server_args,
        "engine_args": engine_args,
    }

    args = {k: v for k, v in args.items() if v is not None}
    args.update(kwargs)

    for model_name in model_names:
        if model_name in CONFIGS:
            raise ValueError(f"Configuration for model '{model_name}' already exists.")

        config = VllmConfig(
            base_model=model_name,
            openai_config=OpenAIServerConfig(**args),
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
        # max_num_seqs=64,
        max_model_len=8192,
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
