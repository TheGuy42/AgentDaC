import argparse
import os

def main():
    """
    Runs the vLLM server with specified arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run the vLLM server on a specific GPU.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="The name or path of the model to serve (e.g., 'meta-llama/Meta-Llama-3-8B-Instruct')."
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="The ID of the GPU to use (e.g., 0)."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="vllm_config.yaml",
        help="Path to the vLLM configuration file (default: vllm_config.yaml)."
    )


    args = parser.parse_args()

    # --- Construct and run the vLLM command ---
    command = (
        "export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True && "
        f"CUDA_VISIBLE_DEVICES={args.gpu} "
        f"vllm serve "
        f" \"{args.model}\" "
        f"--config {args.config} "
        
    )
    # ensure dynamic lora updates are enabled
    print(f"🚀 Running command: {command}")
    os.system(command)

if __name__ == "__main__":
    main()