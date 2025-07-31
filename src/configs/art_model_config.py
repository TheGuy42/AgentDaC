from __future__ import annotations

from art.dev import (
    InternalModelConfig,
    InitArgs,
    EngineArgs,
    PeftArgs,
    TrainerArgs,
    TorchtuneArgs,
)

from pydantic import BaseModel, Field
from art.dev.get_model_config import get_model_config


class ArtConfig(BaseModel, frozen=False, extra="allow"):
    """
    Configuration for an ART model.
    """

    model_name: str
    internal_config: InternalModelConfig = Field(default_factory=InternalModelConfig)

    def to_full(self, output_dir: str) -> ArtConfig:
        internal_config = get_model_config(
            base_model=self.model_name,
            output_dir=output_dir,
            config=self.internal_config,
        )
        self.internal_config = internal_config
        return self


CONFIGS: dict[str, ArtConfig] = {}


def add_config(
    *model_names: str,
    init_args: InitArgs | None = None,
    engine_args: EngineArgs | None = None,
    peft_args: PeftArgs | None = None,
    trainer_args: TrainerArgs | None = None,
    torchtune_args: TorchtuneArgs | None = None,
    **kwargs,
):
    """
    Add a configuration to the global CONFIGS dictionary.
    """

    args = {
        "init_args": init_args,
        "engine_args": engine_args,
        "peft_args": peft_args,
        "trainer_args": trainer_args,
        "torchtune_args": torchtune_args,
    }

    args = {k: v for k, v in args.items() if v is not None}
    args.update(kwargs)

    for model_name in model_names:
        if model_name in CONFIGS:
            raise ValueError(f"Configuration for model '{model_name}' already exists.")

        config = ArtConfig(
            model_name=model_name,
            internal_config=InternalModelConfig(**args),
        )

        CONFIGS[config.model_name] = config


def available_configs() -> list[str]:
    """
    Returns a list of available model configurations.
    """
    return list(CONFIGS.keys())


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
    trainer_args=TrainerArgs(vllm_gpu_memory_utilization=0.7),
)


add_config(
    "Qwen/Qwen2.5-32B-Instruct",
    init_args=InitArgs(
        load_in_4bit=False,
        max_seq_length=10000,
        gpu_memory_utilization=0.85,
    ),
)


add_config(
    "unsloth/Qwen2.5-32B-Instruct",
    init_args=InitArgs(
        load_in_4bit=False,
        max_seq_length=10000,
        gpu_memory_utilization=0.8,
    ),
    trainer_args=TrainerArgs(vllm_gpu_memory_utilization=0.8),
)

# TODO: there are two places with engine_args. The first place is OpenAIServerConfig, which is passed
# to the backend and used to initialize vllm. The second is in InternalModelConfig. What happens when
# these configs disagree with each other? which config is used for what?

add_config(
    "unsloth/Qwen2.5-14B-Instruct",
    init_args=InitArgs(
        load_in_4bit=False,
        max_seq_length=4096,
        gpu_memory_utilization=0.8,
    ),
    engine_args=EngineArgs(
        max_num_batched_tokens=4096*4,
        max_seq_len_to_capture=4096,
        multi_step_stream_outputs=False,
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
    "unsloth/Qwen3-14B",
    init_args=InitArgs(
        load_in_4bit=False,
        max_seq_length=15000,
        gpu_memory_utilization=0.65,
    ),
)

add_config(
    "unsloth/Qwen3-14B-bnb-4bit",
    init_args=InitArgs(
        load_in_4bit=True,
        max_seq_length=20000,
        gpu_memory_utilization=0.65,
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

add_config(
    "unsloth/Llama-4-Scout-17B-16E-Instruct",
    init_args=InitArgs(
        load_in_4bit=False,
        max_seq_length=10000,
        gpu_memory_utilization=0.85,
    ),
)
