from art.dev.model import InitArgs, EngineArgs, PeftArgs, TrainerArgs, InternalModelConfig


generic_32B = InternalModelConfig(
    init_args=InitArgs(
        model_name="",
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

configs:list[InternalModelConfig] = {
    "32B": generic_32B,
}


