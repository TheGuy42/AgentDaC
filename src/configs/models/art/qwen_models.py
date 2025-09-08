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
    openai_config=OpenAIServerConfig(
        server_args=ServerArgs(
            port=8007,
        ),
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
    openai_config=OpenAIServerConfig(
        server_args=ServerArgs(
            port=8001,
        ),
    ),
)