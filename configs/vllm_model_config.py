from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from collections import defaultdict


@dataclass
class VLLMModelConfig:
    """
    Configuration for a vLLM model.
    """

    model_name: str = field(metadata={"help": "The name or path of the model to serve."})
    kwargs: Dict[str, Any] = field(
        default_factory=dict, metadata={"help": "Additional keyword arguments for the vLLM server."}
    )

    def parse_kwargs(self) -> str:
        """
        Converts the kwargs dictionary to a string format suitable for command line arguments.
        """
        return " ".join(f"--{k} {v}" for k, v in self.kwargs.items())

    def run_server(self, port: int = 8200, gpu: str = "1"):
        """
        Constructs the command to run the vLLM server with this model configuration.
        """
        command = (
            "python run_vllm.server.py "
            f"--model {self.model_name} "
            f"--gpu {gpu} "
            f"--port {port} "
            f'kwargs "' + self.parse_kwargs() + '" '
        )

        return command


########################################################################
######################### Model Configurations #########################
########################################################################

######################### Qwen2.5-32B Instruct #########################

Qwen2_5_32B = VLLMModelConfig(
    model_name="Qwen/Qwen2.5-32B-Instruct",
    kwargs={
        "max-num-seqs": 128,
        "max_model_len": 10000,
        "gpu-memory-utilization": 0.85,
    },
)
Qwen2_5_32B_unsloth = VLLMModelConfig(
    model_name="unsloth/Qwen2.5-32B-Instruct",
    kwargs={
        "max-num-seqs": 128,
        "max_model_len": 10000,
        "gpu-memory-utilization": 0.85,
    },
)

######################### Qwen2.5-14B Instruct #########################

Qwen2_5_14B_unsloth = VLLMModelConfig(
    model_name="unsloth/Qwen2.5-14B-Instruct",
    kwargs={
        "max-num-seqs": 128,
        # "max_model_len": 32768,
        # "gpu-memory-utilization": 0.7,
    },
)
######################### Qwen2.5-14B Instruct #########################

Qwen2_5_14B_unsloth_bnb_4bit = VLLMModelConfig(
    model_name="unsloth/Qwen2.5-14B-Instruct-bnb-4bit",
    kwargs={
        "max-num-seqs": 128,
        # "max_model_len": 32768,
        "gpu-memory-utilization": 0.7,
        "load-format": "bitsandbytes",
    },
)

######################### Qwen3-14B Instruct #########################
Qwen3_14B_unsloth_bnb_4bit = VLLMModelConfig(
    model_name="unsloth/Qwen3-14B-bnb-4bit",
    kwargs={
        "max-num-seqs": 128,
        # "max_model_len": 32768,
        "gpu-memory-utilization": 0.7,
        "load-format": "bitsandbytes",
    },
)

######################### Llama-4-Scout 17B 16E Instruct #########################

Llama4_Scout_17B_16E_Instruct_bnb_4bit = VLLMModelConfig(
    model_name="unsloth/Llama-4-Scout-17B-16E-Instruct-unsloth-dynamic-bnb-4bit",
    kwargs={
        "max-num-seqs": 128,
        # "max_model_len": 10000,
        "gpu-memory-utilization": 0.85,
        "load-format": "bitsandbytes",
    },
)


######################### Configurations Dictionary #########################

model_configs: Dict[str, VLLMModelConfig] = {
    Qwen2_5_32B.model_name: Qwen2_5_32B,
    Qwen2_5_32B_unsloth.model_name: Qwen2_5_32B_unsloth,
    Qwen2_5_14B_unsloth.model_name: Qwen2_5_14B_unsloth,
    Qwen2_5_14B_unsloth_bnb_4bit.model_name: Qwen2_5_14B_unsloth_bnb_4bit,
    Qwen3_14B_unsloth_bnb_4bit.model_name: Qwen3_14B_unsloth_bnb_4bit,
    Llama4_Scout_17B_16E_Instruct_bnb_4bit.model_name: Llama4_Scout_17B_16E_Instruct_bnb_4bit,
}
