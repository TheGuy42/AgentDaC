from art.dev.model import (
    InternalModelConfig,
    InitArgs,
    EngineArgs,
    PeftArgs,
    TrainerArgs,
    TorchtuneArgs,
)

import torch


# TODO: we do not need to write the entire config here, only the parameters
# which we wish to change from the default values.

########################################################################
######################### Model Configurations #########################
########################################################################

######################### Qwen2-7B Instruct #########################
Qwen2_7B_unsloth = InternalModelConfig(
    init_args=InitArgs(
        model_name="unsloth/Qwen2-7B",
        max_seq_length=32768,
        load_in_4bit=False,
        gpu_memory_utilization=0.7,  # Reduce if out of memory
        max_lora_rank=8,
    ),
    trainer_args=TrainerArgs(
        vllm_gpu_memory_utilization=0.7,  # Reduce if out of memory
    ),
    # engine_args=EngineArgs(
    #     gpu_memory_utilization=0.7,  # Reduce if out of memory
    #     max_model_len= 32768,  # Ensure this matches the model's max sequence length
    # ),
    # engine_args=EngineArgs(
    #     # pipeline_parallel_size=2,
    #     tensor_parallel_size=2,
    #     # enable_sleep_mode=False,
    # ),
    # torchtune_args=TorchtuneArgs(
    #     model="qwen2_7b",
    #     model_type="QWEN2",
    #     async_weight_syncing=True,
    # )
)


######################### Qwen2.5-32B Instruct #########################
Qwen2_5_32B = InternalModelConfig(
    init_args=InitArgs(
        model_name="Qwen/Qwen2.5-32B-Instruct",
        max_seq_length=10000,
        load_in_4bit=False,
        gpu_memory_utilization=0.85,  # Reduce if out of memory
        max_lora_rank=8,
    )
)
Qwen2_5_32B_unsloth = InternalModelConfig(
    init_args=InitArgs(
        model_name="unsloth/Qwen2.5-32B-Instruct",
        max_seq_length=10000,
        load_in_4bit=False,
        gpu_memory_utilization=0.8,  # Reduce if out of memory
        max_lora_rank=8,
    ),
    trainer_args=TrainerArgs(
        vllm_gpu_memory_utilization=0.8,  # Reduce if out of memory
    ),
)

######################### Qwen2.5-14B Instruct #########################


# NOTE: works, trains and runs inference on a single GPU
Qwen2_5_14B_unsloth = InternalModelConfig(
    init_args=InitArgs(
        model_name="unsloth/Qwen2.5-14B-Instruct",
        max_seq_length=10000,
        load_in_4bit=False,
        gpu_memory_utilization=0.7,
        max_lora_rank=8,
    ),
)

# NOTE: potentially working tensor parallel config
# Qwen2_5_14B_unsloth = InternalModelConfig(
#     init_args=InitArgs(
#         model_name="unsloth/Qwen2.5-14B-Instruct",
#         # max_seq_length=10000,
#         load_in_4bit=False,
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
# Qwen2_5_14B_unsloth = InternalModelConfig(
#     init_args=InitArgs(
#         model_name="unsloth/Qwen2.5-14B-Instruct",
#         max_seq_length=10000,
#         load_in_4bit=False,
#         gpu_memory_utilization=0.7,
#         max_lora_rank=8,
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

Qwen2_5_14B_unsloth_bnb_4bit = InternalModelConfig(
    init_args=InitArgs(
        model_name="unsloth/Qwen2.5-14B-Instruct-bnb-4bit",
        max_seq_length=32768,
        load_in_4bit=True,
        gpu_memory_utilization=(0.5),  # Reduce if out of memory
        max_lora_rank=8,
    ),
    # trainer_args=TrainerArgs(
    #     vllm_gpu_memory_utilization=0.7,  # Reduce if out of memory
    # ),
    # engine_args=EngineArgs(
    #     gpu_memory_utilization=0.7,  # Reduce if out of memory
    #     max_model_len= 32768,  # Ensure this matches the model's max sequence length
    # ),
    # engine_args=EngineArgs(
    #     # pipeline_parallel_size=2,
    #     tensor_parallel_size=2,
    #     # enable_sleep_mode=False,
    # ),
    # torchtune_args=TorchtuneArgs(
    #     model="qwen2_5_14b_instruct",
    #     model_type="QWEN2",
    #     async_weight_syncing=True,
    # )
)

######################### Qwen3-14B Instruct #########################
Qwen3_14B_unsloth = InternalModelConfig(
    init_args=InitArgs(
        model_name="unsloth/Qwen3-14B",
        max_seq_length=15000,
        load_in_4bit=True,
        gpu_memory_utilization=(0.65),  # Reduce if out of memory
        max_lora_rank=8,
    ),
    # trainer_args=TrainerArgs(
    #     # vllm_gpu_memory_utilization=(0.9),  # Reduce if out of memory
    #     num_generations=1,
    #     per_device_train_batch_size=1,
    # ),
    # engine_args=EngineArgs(
    #     pipeline_parallel_size=2,
    #     # tensor_parallel_size=2,
    #     # enable_sleep_mode=False,
    #     # gpu_memory_utilization=(0.9),  # Reduce if out of memory
    # ),
    # torchtune_args=TorchtuneArgs(
    #     model="qwen3_14b_instruct",
    #     model_type="QWEN3",
    #     async_weight_syncing=False,
    #     enable_activation_offloading=True,
    # )
)

Qwen3_14B_unsloth_bnb_4bit = InternalModelConfig(
    init_args=InitArgs(
        model_name="unsloth/Qwen3-14B-bnb-4bit",
        max_seq_length=20000,
        load_in_4bit=True,
        gpu_memory_utilization=0.65,
        max_lora_rank=8,
    ),
)

######################### Llama-4-Scout 17B 16E Instruct #########################
Llama4_Scout_17B_16E_Instruct_bnb_4bit = InternalModelConfig(
    init_args=InitArgs(
        model_name="unsloth/Llama-4-Scout-17B-16E-Instruct-unsloth-dynamic-bnb-4bit",
        max_seq_length=10000,
        load_in_4bit=True,
        gpu_memory_utilization=0.85,  # Reduce if out of memory
        max_lora_rank=8,
    ),
    # engine_args=EngineArgs(
    #     pipeline_parallel_size=2,
    # ),
    # torchtune_args=TorchtuneArgs(
    #     model="llama4_scout_17b_16e",
    #     model_type="LLAMA4",
    #     async_weight_syncing=True,
    # )
)

Llama4_Scout_17B_16E_Instruct = InternalModelConfig(
    init_args=InitArgs(
        model_name="unsloth/Llama-4-Scout-17B-16E-Instruct",
        max_seq_length=10000,
        load_in_4bit=True,
        gpu_memory_utilization=0.85,  # Reduce if out of memory
        max_lora_rank=8,
    ),
    # engine_args=EngineArgs(
    #     pipeline_parallel_size=3,
    # ),
    # torchtune_args=TorchtuneArgs(
    #     model="llama4_scout_17b_16e",
    #     model_type="LLAMA4",
    #     async_weight_syncing=True,
    # )
)


######################### Configurations Dictionary #########################
configs: dict[str, InternalModelConfig] = {
    "32B": Qwen2_5_32B,
    Qwen2_5_32B["init_args"]["model_name"]: Qwen2_5_32B,
    Qwen2_5_32B_unsloth["init_args"]["model_name"]: Qwen2_5_32B_unsloth,
    Qwen2_5_14B_unsloth["init_args"]["model_name"]: Qwen2_5_14B_unsloth,
    Qwen2_5_14B_unsloth_bnb_4bit["init_args"]["model_name"]: Qwen2_5_14B_unsloth_bnb_4bit,
    Qwen3_14B_unsloth["init_args"]["model_name"]: Qwen3_14B_unsloth,
    Qwen3_14B_unsloth_bnb_4bit["init_args"]["model_name"]: Qwen3_14B_unsloth_bnb_4bit,
    Llama4_Scout_17B_16E_Instruct_bnb_4bit["init_args"]["model_name"]: Llama4_Scout_17B_16E_Instruct_bnb_4bit,
    Llama4_Scout_17B_16E_Instruct["init_args"]["model_name"]: Llama4_Scout_17B_16E_Instruct,
}
