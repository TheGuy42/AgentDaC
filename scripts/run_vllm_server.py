import logging
import torch
import argparse
import os
import sys
import pathlib

# set pythonpath to the main module directory
module_dir = pathlib.Path(__file__).parent.resolve().parent
if str(module_dir) not in sys.path:
    sys.path.append(str(module_dir))

from src.models import load_vllm_model
from src.utils.env import prepare_environment
from src.utils.logging import setup_logging
from src.configs.vllm_model_config import available_configs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the vLLM server on a specific GPU.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help=f"The name or path of the model to serve. Available models are: {available_configs()}",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8200,
        help="The port on which the vLLM server will run (e.g., '8000').",
    )

    parser.add_argument(
        "--gpu",
        type=int,
        nargs="+",
        default=[0],
        help=f"The ID of the GPU(s) to use (e.g., 0 or '0,1'). Available GPUs: {list(range(torch.cuda.device_count()))}",
    )

    parser.add_argument(
        "--kwargs",
        type=str,
        default="",
        help="Additional keyword arguments to pass to the vLLM server.",
    )

    return parser.parse_args()


def main(args: argparse.Namespace):
    prepare_environment()

    vllm_args = load_vllm_model(model_name=args.model, port=args.port)

    gpus_string = ",".join([str(gpu) for gpu in args.gpu])

    command = (
        f"export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True && "
        f"CUDA_VISIBLE_DEVICES={gpus_string} && "
        # f"VLLM_ATTENTION_BACKEND=FLASH_ATTN && " # TODO: does modifying attn backend can significantly improve performance?
        # f"VLLM_FLASH_ATTN_VERSION=2 "
        f"vllm serve {' '.join(vllm_args)} {args.kwargs} "
    )

    print()
    print(f"🚀 Running command: {command}")
    print()

    os.system(command)


if __name__ == "__main__":
    setup_logging(logging.INFO)
    args = parse_args()
    main(args)
