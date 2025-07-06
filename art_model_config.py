from art.dev.model import InitArgs, EngineArgs, PeftArgs, TrainerArgs, InternalModelConfig
from typing import Dict

########################################################################
######################### Model Configurations #########################
########################################################################

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
    )
)
######################### Qwen2.5-14B Instruct #########################
Qwen2_5_14B_unsloth = InternalModelConfig(
    init_args=InitArgs(
        model_name="unsloth/Qwen2.5-14B-Instruct",
        max_seq_length=10000,
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
    engine_args=EngineArgs(
        gpu_memory_utilization=(0.7),  # Reduce if out of memory
        max_model_len= 10000,  # Ensure this matches the model's max sequence length
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
    )
)


######################### Configurations Dictionary #########################
configs:Dict[str, InternalModelConfig] = {
    "32B": Qwen2_5_32B,
    Qwen2_5_32B["init_args"]["model_name"]: Qwen2_5_32B,
    Qwen2_5_32B_unsloth["init_args"]["model_name"]: Qwen2_5_32B_unsloth,
    Qwen2_5_14B_unsloth["init_args"]["model_name"]: Qwen2_5_14B_unsloth,
    Llama4_Scout_17B_16E_Instruct_bnb_4bit["init_args"]["model_name"]: Llama4_Scout_17B_16E_Instruct_bnb_4bit,
}


