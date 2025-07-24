from art.dev.model import InitArgs, EngineArgs, PeftArgs, TrainerArgs, InternalModelConfig
from art.local import LocalBackend
import argparse
import os
import re

from src.utils.api_key_utils import api_key_from_file
from easy2hard.trainer import Easy2HardTrainer
from src.configs.art_model_config import configs


def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Run the Easy2Hard training process.")

    parser.add_argument(
        "--model_name",
        type=str,
        help="The name of the model to use for training.",
        required=True,
    )
    # parser.add_argument(
    #     "--model_config",
    #     type=str,
    #     default="32B",
    #     help="The name of the model configuration to use."
    # )
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs to train the model.")
    parser.add_argument("--n_rollouts", type=int, default=10, help="Number of rollouts per group.")
    parser.add_argument("--n_groups", type=int, default=5, help="Number of groups to use for training.")
    parser.add_argument(
        "--gpu",
        type=int,
        nargs="+",
        default=[0],  # Default to GPU 0
        help="GPU ID to use for training.",
    )
    parser.add_argument(
        "--vllm_server_ports",
        type=int,
        nargs="+",
        default=[],  # Default port for vLLM server
        help="List of ports for the vLLM servers.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Name of the run. If provided, it will be used to resume the training from the given run name.",
    )

    args = parser.parse_args()
    return args


async def main():
    """
    Main function to run the training process.
    """
    os.environ["TORCHINDUCTOR_MAX_AUTOTUNE"] = "1"
    WANDB_API_KEY = api_key_from_file("api_keys/WANDB_KEY.txt")
    args = parse_args()
    print("Running training with the following arguments:")
    print(args)

    model_config: InternalModelConfig = configs.get(args.model_name, None)
    
    if model_config is None:
        raise ValueError(f"Model configuration for {args.model_name} not found in configs.")

    # Initialize the trainer
    trainer = Easy2HardTrainer(
        model_name=args.model_name,
        model_config=model_config,
        run_name=args.run_name,
        backend=LocalBackend(in_process=True, path="./.art"),
        WANDB_API_KEY=WANDB_API_KEY,
        seed=42,
        gpu=args.gpu,
        vllm_server_ports=args.vllm_server_ports,
    )

    # Load the model
    await trainer.load_model()  # Example port, adjust as necessary

    wandb = trainer.get_wandb_run()
    if wandb is not None:
        files_to_log = [
            "easy2hard/trainer.py",
            "easy2hard/run_exp.py",
            "src/configs/art_model_config.py",
            "src/configs/vllm_model_config.py",
            "src/dac_agent.py",
            "src/sys_prompt.py",
            "src/vllm_client.py",
            "src/training.py",
        ]
        wandb.log_code(
            include_fn=lambda path: any(re.search(file, path) for file in files_to_log),
        )
        trainer.update_wandb_config(args.__dict__)

    # Start training
    await trainer.train(
        epochs=args.epochs,
        n_rollouts=args.n_rollouts,
        n_groups=args.n_groups,
    )


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
    # Note: Ensure that the vLLM servers are running on the specified ports before starting the training process.
    # You can start the vLLM servers using the `run_vllm.server.py` script with the appropriate model configurations.
