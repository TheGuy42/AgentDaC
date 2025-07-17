from art.dev.model import InitArgs, EngineArgs, PeftArgs, TrainerArgs, InternalModelConfig, TorchtuneArgs
from typing import Dict

########################################################################
######################### Model Configurations #########################
########################################################################

######################### Qwen2-7B Instruct #########################
Qwen2_7B_unsloth = InternalModelConfig(
    init_args=InitArgs(
        model_name="unsloth/Qwen2-7B",
        max_seq_length=32768,
        load_in_4bit=False,  # False for LoRA 16bit
        fast_inference=True,  # Enable vLLM fast inference
        # vLLM args
        disable_log_stats=False,
        enable_prefix_caching=True,
        gpu_memory_utilization=(0.7),  # Reduce if out of memory
        max_lora_rank=8,
        use_async=True,
    ),
    trainer_args=TrainerArgs(
        vllm_gpu_memory_utilization=(0.7),  # Reduce if out of memory
    ),
    # engine_args=EngineArgs(
    #     gpu_memory_utilization=(0.7),  # Reduce if out of memory
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
        load_in_4bit=False,  # False for LoRA 16bit
        fast_inference=True,  # Enable vLLM fast inference
        # vLLM args
        disable_log_stats=False,
        enable_prefix_caching=True,
        gpu_memory_utilization=(0.85),  # Reduce if out of memory
        max_lora_rank=8,
        use_async=True,
    )
)
Qwen2_5_32B_unsloth = InternalModelConfig(
    init_args=InitArgs(
        model_name="unsloth/Qwen2.5-32B-Instruct",
        max_seq_length=10000,
        load_in_4bit=False,  # False for LoRA 16bit
        fast_inference=True,  # Enable vLLM fast inference
        # vLLM args
        disable_log_stats=False,
        enable_prefix_caching=True,
        gpu_memory_utilization=(0.8),  # Reduce if out of memory
        max_lora_rank=8,
        use_async=True,
    ),
    trainer_args=TrainerArgs(
        vllm_gpu_memory_utilization=(0.8),  # Reduce if out of memory
    ),
    
)

######################### Qwen2.5-14B Instruct #########################
Qwen2_5_14B_unsloth = InternalModelConfig(
    init_args=InitArgs(
        model_name="unsloth/Qwen2.5-14B-Instruct",
        max_seq_length=32768,
        load_in_4bit=False,  # False for LoRA 16bit
        fast_inference=True,  # Enable vLLM fast inference
        # vLLM args
        disable_log_stats=False,
        enable_prefix_caching=True,
        gpu_memory_utilization=(0.7),  # Reduce if out of memory
        max_lora_rank=8,
        use_async=True,
    ),
    trainer_args=TrainerArgs(
        vllm_gpu_memory_utilization=(0.7),  # Reduce if out of memory
    ),
    # engine_args=EngineArgs(
    #     gpu_memory_utilization=(0.7),  # Reduce if out of memory
    #     max_model_len= 32768,  # Ensure this matches the model's max sequence length
    # ),
    # engine_args=EngineArgs(
    #     # pipeline_parallel_size=2,
    #     tensor_parallel_size=3,
    #     # enable_sleep_mode=False,
    # ),
    # torchtune_args=TorchtuneArgs(
    #     model="qwen2_5_14b_instruct",
    #     model_type="QWEN2",
    #     async_weight_syncing=True,
    # )
)

Qwen2_5_14B_unsloth_bnb_4bit = InternalModelConfig(
    init_args=InitArgs(
        model_name="unsloth/Qwen2.5-14B-Instruct-bnb-4bit",
        max_seq_length=32768,
        load_in_4bit=True,  # False for LoRA 16bit
        fast_inference=True,  # Enable vLLM fast inference
        # vLLM args
        disable_log_stats=False,
        enable_prefix_caching=True,
        gpu_memory_utilization=(0.5),  # Reduce if out of memory
        max_lora_rank=8,
        use_async=True,
    ),
    # trainer_args=TrainerArgs(
    #     vllm_gpu_memory_utilization=(0.7),  # Reduce if out of memory
    # ),
    # engine_args=EngineArgs(
    #     gpu_memory_utilization=(0.7),  # Reduce if out of memory
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
        load_in_4bit=True,  # False for LoRA 16bit
        fast_inference=True,  # Enable vLLM fast inference
        # vLLM args
        disable_log_stats=False,
        enable_prefix_caching=True,
        gpu_memory_utilization=(0.65),  # Reduce if out of memory
        max_lora_rank=8,
        use_async=True,
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
        load_in_4bit=True,  # False for LoRA 16bit
        fast_inference=True,  # Enable vLLM fast inference
        # vLLM args
        disable_log_stats=False,
        enable_prefix_caching=True,
        gpu_memory_utilization=(0.65),  # Reduce if out of memory
        max_lora_rank=8,
        use_async=True,
    ),
)

######################### Llama-4-Scout 17B 16E Instruct #########################
Llama4_Scout_17B_16E_Instruct_bnb_4bit = InternalModelConfig(
    init_args=InitArgs(
        model_name="unsloth/Llama-4-Scout-17B-16E-Instruct-unsloth-dynamic-bnb-4bit",
        max_seq_length=10000,
        load_in_4bit=True,  # False for LoRA 16bit
        fast_inference=True,  # Enable vLLM fast inference
        # vLLM args
        disable_log_stats=False,
        enable_prefix_caching=True,
        gpu_memory_utilization=(0.85),  # Reduce if out of memory
        max_lora_rank=8,
        use_async=True,
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
        load_in_4bit=True,  # False for LoRA 16bit
        fast_inference=True,  # Enable vLLM fast inference
        # vLLM args
        disable_log_stats=False,
        enable_prefix_caching=True,
        gpu_memory_utilization=(0.85),  # Reduce if out of memory
        max_lora_rank=8,
        use_async=True,
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
configs:Dict[str, InternalModelConfig] = {
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


