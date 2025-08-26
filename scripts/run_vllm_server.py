import logging
import torch
import argparse
import os
import sys
import pathlib
import subprocess

# set pythonpath to the main module directory
module_dir = pathlib.Path(__file__).parent.resolve().parent
if str(module_dir) not in sys.path:
    sys.path.append(str(module_dir))

from src.utils.loaders import load_vllm_model
from src.utils.env import prepare_environment
from src.utils.logging import create_logger, setup_logging
from src.configs.models.vllm import available_configs, VllmConfig


logger = create_logger(__name__)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
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
        help="The port on which the vLLM server will run (e.g., '8200').",
    )

    parser.add_argument(
        "--gpu",
        type=int,
        nargs="+",
        default=[0],
        help=f"The ID of the GPU(s) to use (e.g., 0 or 0 1).  Available GPUs: {list(range(torch.cuda.device_count()))}",
    )

    parser.add_argument(
        "--vllm_config",
        type=str,
        default=None,
        help="Path to a JSON file containing additional vLLM configuration parameters. "
        "The serialized object should be of type `VllmConfig`.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=69420,
        help="Random seed for reproducibility.",
    )

    args, extra_args = parser.parse_known_args()

    # print the parsed arguments

    print()
    print("Parsed arguments:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print()

    print("Extra arguments (passed to vllm serve):")
    print(extra_args)
    print()

    return args, extra_args


def main(args: argparse.Namespace, extra_args: list[str]) -> None:
    print()
    print(f"Current working directory: {os.getcwd()}")
    print()

    vllm_config = None
    if args.vllm_config is not None:
        with open(args.vllm_config, "r") as f:
            vllm_config = VllmConfig.model_validate_json(f.read())

    vllm_args = load_vllm_model(
        model_name=args.model,
        port=args.port,
        seed=args.seed,
        vllm_config=vllm_config,
        print_full=True,
    )

    env = os.environ.copy()
    env.update(
        {
            "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "True",
            "CUDA_VISIBLE_DEVICES": ",".join([str(gpu) for gpu in args.gpu]),
            # "VLLM_ATTENTION_BACKEND": "FLASH_ATTN",   # TODO: does modifying attn backend can significantly improve performance?
            # "VLLM_FLASH_ATTN_VERSION": "3", # TODO: comment / uncomment when necessary
        }
    )

    cmd_args = ["vllm", "serve"] + vllm_args + extra_args

    print()
    print(f"🚀 Running command: {' '.join(cmd_args)}")
    print()

    try:
        subprocess.run(cmd_args, env=env, shell=False)
    except KeyboardInterrupt:
        pass  # Allow graceful shutdown on Ctrl+C


if __name__ == "__main__":
    prepare_environment()
    setup_logging(logging.WARNING)
    args, extra_args = parse_args()
    main(args, extra_args)
