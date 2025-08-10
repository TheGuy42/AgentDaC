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

from pydantic import BaseModel, Field, model_validator
from pathlib import Path
from art.dev.get_model_config import get_model_config
from src.utils.io import save_base_model


class ArtConfig(BaseModel, frozen=False, extra="allow"):
    """
    Configuration for an ART model.
    """

    id: str = ""  # NOTE: not supported yet
    base_model: str
    internal_config: InternalModelConfig = Field(default_factory=InternalModelConfig)
    openai_config: OpenAIServerConfig | None = None

    @model_validator(mode="after")
    def validate_identifier(self) -> ArtConfig:
        """
        Validate that the identifier is set to the base model name if not provided.
        """
        if not self.id:
            self.id = self.base_model
        return self

    def initialize(self, output_dir: str) -> ArtConfig:
        self.internal_config = get_model_config(
            base_model=self.base_model,
            output_dir=output_dir,
            config=self.internal_config,
        )
        self.internal_config["engine_args"].setdefault("seed", 0)  # type: ignore
        return self

    def save(self, dir_name: str, file_name: str = "art_config.json") -> None:
        save_base_model(self, Path(dir_name) / file_name)


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

# TODO: there is an option to load in 8bit, its also a quantization, test it.

add_config(
    "unsloth/Qwen2-7B",
    init_args=InitArgs(
        load_in_4bit=False,
        gpu_memory_utilization=0.7,
    ),
)

add_config(
    "unsloth/Qwen2.5-14B-Instruct",
    init_args=InitArgs(
        load_in_4bit=False,
        max_seq_length=4096 * 2,
        gpu_memory_utilization=0.8,
    ),
    engine_args=EngineArgs(
        max_num_batched_tokens=4096 * 4 * 2,
        max_seq_len_to_capture=4096 * 2,
    ),
    openai_config=OpenAIServerConfig(
        server_args=ServerArgs(
            port=8001,
        ),
    ),
)

# NOTE: potentially working tensor parallel config
# add_config(
#     "unsloth/Qwen2.5-14B-Instruct",
#     init_args=InitArgs(
#         # max_seq_length=10000,
#         # device_map="cuda:0",
#         # gpu_memory_utilization=0.7,
#         # max_lora_rank=8,
#     ),
#     engine_args=EngineArgs(
#         tensor_parallel_size=2,
#         gpu_memory_utilization=0.5,
#         max_model_len=10000,
#     ),
#     _decouple_vllm_and_unsloth=True, # Must be used for multi-gpu support
# )

# NOTE: potentially working tensor parallel config
# It crashes during training step, potentially due to OOM
# add_config(
#     "unsloth/Qwen2.5-14B-Instruct",
#     init_args=InitArgs(
#         max_seq_length=10000,
#         gpu_memory_utilization=0.7,
#     ),
#     torchtune_args=TorchtuneArgs(
#         model="qwen2_5_14b_instruct",
#         model_type="QWEN2",
#     ),
#     engine_args=EngineArgs(
#         tensor_parallel_size=2,
#         gpu_memory_utilization=0.7,
#         max_model_len=5000,
#         max_num_batched_tokens=5000,
#     ),
# )

add_config(
    "unsloth/Qwen2.5-14B-Instruct-bnb-4bit",
    init_args=InitArgs(
        load_in_4bit=True,
        gpu_memory_utilization=0.5,
    ),
)

add_config(
    "unsloth/Qwen2.5-32B-Instruct",
    init_args=InitArgs(
        load_in_4bit=False,
        max_seq_length=10000,
        gpu_memory_utilization=0.8,
    ),
)

add_config(
    "unsloth/Qwen3-14B",
    init_args=InitArgs(
        load_in_4bit=False,
        gpu_memory_utilization=0.8,
    ),
    engine_args=EngineArgs(
        # max_num_batched_tokens=1024 * 64,
        # max_seq_len_to_capture=1024 * 16,
        gpu_memory_utilization=0.8,
    ),
    openai_config=OpenAIServerConfig(
        server_args=ServerArgs(
            port=8001,
        ),
    ),
)

add_config(
    "unsloth/Llama-4-Scout-17B-16E-Instruct",
    init_args=InitArgs(
        load_in_4bit=False,
        max_seq_length=10000,
        gpu_memory_utilization=0.85,
    ),
)


add_config(
    "unsloth/Llama-4-Scout-17B-16E-Instruct-unsloth-dynamic-bnb-4bit",
    init_args=InitArgs(
        load_in_4bit=True,
        max_seq_length=10000,
        gpu_memory_utilization=0.85,
    ),
)
