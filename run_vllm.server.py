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
        "--port",
        type=int,
        default=8200,
        help="The port on which the vLLM server will run (e.g., '8000')."
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        help="The ID of the GPU(s) to use (e.g., 0 or '0,1')."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="vllm_config.yaml",
        help="Path to the vLLM configuration file (default: vllm_config.yaml)."
    )

    parser.add_argument(
        "--kwargs",
        type=str,
        default="",
        help="Additional keyword arguments to pass to the vLLM server."
    )


    args = parser.parse_args()

    # --- Construct and run the vLLM command ---
    command = (
        "export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True && "
        f"CUDA_VISIBLE_DEVICES={args.gpu} "
        f"vllm serve "
        f" \"{args.model}\" "
        f"--port {args.port} "
        f"--config {args.config} "
        f"{args.kwargs} "
    )
    # ensure dynamic lora updates are enabled
    print(f"🚀 Running command: {command}")
    os.system(command)

if __name__ == "__main__":
    main()