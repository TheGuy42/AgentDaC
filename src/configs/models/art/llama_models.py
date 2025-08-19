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