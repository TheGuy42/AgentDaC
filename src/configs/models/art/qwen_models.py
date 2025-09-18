from src.configs.models.art import add_config
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
)

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
        gpu_memory_utilization=0.9,
        max_lora_rank=16,
    ),
    engine_args=EngineArgs(
        max_model_len=1024 * 4,
        gpu_memory_utilization=0.9,
        max_lora_rank=16,
    ),
    peft_args=PeftArgs(
        r=16,
    ),
)

add_config(
    "unsloth/Qwen3-32B",
    init_args=InitArgs(
        load_in_4bit=False,
        gpu_memory_utilization=0.9,
        max_lora_rank=16,
    ),
    engine_args=EngineArgs(
        max_model_len=1024 * 8,
        gpu_memory_utilization=0.9,
        max_lora_rank=16,
    ),
    peft_args=PeftArgs(
        r=16,
    ),
)