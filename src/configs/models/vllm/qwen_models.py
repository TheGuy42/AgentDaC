from src.configs.models.vllm import add_config
from art.dev import ServerArgs, EngineArgs, OpenAIServerConfig


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
        max_model_len=1024 * 16,
        gpu_memory_utilization=0.9,
        max_lora_rank=64,
    ),
)

add_config(
    "unsloth/Qwen3-32B",
    engine_args=EngineArgs(
        gpu_memory_utilization=0.9,
    ),
)

add_config(
    "unsloth/Qwen3-32B-unsloth-bnb-4bit",
    engine_args=EngineArgs(
        gpu_memory_utilization=0.95,
    ),
)
