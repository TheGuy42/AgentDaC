# import unsloth
from art.dev.model import InitArgs, EngineArgs, PeftArgs, TrainerArgs, InternalModelConfig
import argparse
import os
import art

from trainer import Easy2HardTrainer
from art_model_config import configs

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
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of epochs to train the model."
    )
    parser.add_argument(
        "--n_rollouts",
        type=int,
        default=10,
        help="Number of rollouts per group."
    )
    parser.add_argument(
        "--n_groups",
        type=int,
        default=5,
        help="Number of groups to use for training."
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU ID to use for training."
    )
    parser.add_argument(
        "--vllm_server_ports",
        type=int,
        nargs='+',
        default=[],  # Default port for vLLM server
        help="List of ports for the vLLM servers."
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Name of the run. If provided, it will be used to resume the training from the given run name."
    )
    
    args = parser.parse_args()
    return args


async def main():
    """
    Main function to run the training process.
    """
    WANDB_API_KEY = "308e1be7938bf7a7c366afc0522fb9fc0d8cf1ad"
    args = parse_args()
    print("Running training with the following arguments:")
    print(args)
    
    model_config:InternalModelConfig = configs.get(args.model_name, None)

    # Initialize the trainer
    trainer = Easy2HardTrainer(
        model_name=args.model_name,
        model_config=model_config,
        run_name=args.run_name,
        WANDB_API_KEY=WANDB_API_KEY,
        seed=42,
        gpu=args.gpu,
        vllm_server_ports=args.vllm_server_ports
    )

    # Load the model
    await trainer.load_model()  # Example port, adjust as necessary

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
    