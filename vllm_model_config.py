from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class VLLMModelConfig:
    """
    Configuration for a vLLM model.
    """
    model_name: str = field(
        metadata={"help": "The name or path of the model to serve."}
    )
    kwargs: Dict[str, Any] = field(
        default_factory=dict,
        metadata={"help": "Additional keyword arguments for the vLLM server."}
    )

    def run_server(self, port: int = 8200, gpu: str = "1"):
        """
        Constructs the command to run the vLLM server with this model configuration.
        """
        command = (
            "python run_vllm.server.py "
            f"--model {self.model_name} "
            f"--gpu {gpu} "
            f"--port {port} "
            f"kwargs \"" + f"--{k} {v} " for k,v in self.kwargs.items() + "\" "
        )
        
        return command


Qwen2_5_32B = VLLMModelConfig(
    model_name="Qwen/Qwen2.5-32B-Instruct",
    kwargs={
        "max-num-seqs": 128,
        "max_model_len": 10000,
        "gpu-memory-utilization": 0.85,
        }
)
















