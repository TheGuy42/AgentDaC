from src.configs.models.vllm import add_config
from art.dev import ServerArgs, EngineArgs, OpenAIServerConfig

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
