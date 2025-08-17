import sys
import os
import torch
import pathlib
import asyncio
import argparse
import logging
from typing import Any

from datasets import Dataset, load_dataset, DatasetDict
import art

# set pythonpath to the main module directory
module_dir = pathlib.Path(__file__).parent.resolve().parent
if str(module_dir) not in sys.path:
    sys.path.append(str(module_dir))

from src.utils.env import prepare_environment
from src.utils.logging import setup_logging
from src.utils.io import load_base_model
from src.models import load_art_model, PathConfig
from src.vllm_client import VllmClient, ArtClient, VllmRouter
from src.trainer import TrainingConfig, PromptConfig, StopCriteria
from src.configs.art_configs import available_configs, ArtConfig
from experiments.easy2hard_ruler.trainer import Easy2HardRulerTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Easy2Hard experiment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help=f"The name or path of the model to serve. Available models are: {available_configs()}",
    )

    parser.add_argument(
        "--project",
        type=str,
        default="easy2hard_dac_ruler",
        help="The name of the project for saving results.",
    )

    parser.add_argument(
        "--run",
        type=str,
        default="",
        help="The name of the run to continue (if exists).",
    )

    parser.add_argument(
        "--gpu",
        type=int,
        nargs="+",
        default=[0],
        help=f"The ID of the GPU(s) to use (e.g., 0 or 0 1). Available GPUs: {list(range(torch.cuda.device_count()))}",
    )

    parser.add_argument(
        "--vllm_ports",
        type=int,
        nargs="*",
        default=[],
        help="List of endpoint ports for vLLM servers.",
    )

    parser.add_argument(
        "--config_dir",
        type=str,
        default="experiments/easy2hard_ruler/defaults",
        help="Directory containing experiment configuration files.",
    )

    args = parser.parse_args()

    # verify valid GPU IDs
    if not all(0 <= gpu < torch.cuda.device_count() for gpu in args.gpu):
        raise ValueError(
            f"Invalid GPU IDs provided: {args.gpu}. Available GPUs: {list(range(torch.cuda.device_count()))}"
        )

    # print the parsed arguments
    print("Parsed arguments:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")

    return args


def load_data() -> tuple[Dataset, Dataset]:
    dataset_dict: DatasetDict = load_dataset(
        path="furonghuang-lab/Easy2Hard-Bench",
        name="E2H-AMC",
        split=None,
    )  # type: ignore

    ds_train: Dataset = dataset_dict["train"]
    ds_val: Dataset = dataset_dict["eval"]
    return ds_train, ds_val


def load_configs(config_dir: str | pathlib.Path) -> dict[str, Any]:
    if isinstance(config_dir, str):
        config_dir = pathlib.Path(config_dir)

    return {
        "art_config": load_base_model(ArtConfig, config_dir / "art_config.json", do_raise=False),
        "train_config": load_base_model(TrainingConfig, config_dir / "train_config.json", do_raise=True),
        "prompt_config": load_base_model(PromptConfig, config_dir / "prompt_config.json", do_raise=True),
        "stop_criteria": load_base_model(StopCriteria, config_dir / "stop_criteria.json", do_raise=True),
    }


async def main(args: argparse.Namespace):
    """
    Main function to run the training process.
    """
    # Set the GPU environment variable
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, args.gpu))

    print()
    print(f"Current working directory: {os.getcwd()}")
    print()

    path_config = PathConfig(
        base_model=args.model,
        project_name=args.project,
        run_name=args.run,
    )

    # Load configurations
    config_dict = load_configs(args.config_dir)
    art_config: ArtConfig | None = config_dict["art_config"]
    train_config: TrainingConfig = config_dict["train_config"]
    prompt_config: PromptConfig = config_dict["prompt_config"]
    stop_criteria: StopCriteria = config_dict["stop_criteria"]

    if "Qwen3" in args.model:
        # Automatically disable thinking for Qwen3 models
        train_config.rollout_kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}

    # load model
    model = await load_art_model(
        path_config=path_config,
        art_config=art_config,
        print_full=True,
    )

    # load dataset
    ds_train, ds_val = load_data()

    # create inference clients
    inference_clients = [ArtClient.from_art_model(model)]
    for port in args.vllm_ports:
        inference_clients.append(
            VllmClient.from_connection(
                port=port,
                base_model=model.base_model,
                model_name=model.get_inference_name(),
            )
        )

    vllm_router = VllmRouter(inference_clients)

    # create and configure the trainer
    trainer = Easy2HardRulerTrainer(
        model=model,
        vllm_router=vllm_router,
        path_config=path_config,
        prompt_config=prompt_config,
        stop_criteria=stop_criteria,
    )

    # log code files
    if trainer.wandb_run is not None:
        trainer.wandb_run.log_code(root="src", name="src")
        trainer.wandb_run.log_code(root="scripts", name="scripts")
        trainer.wandb_run.log_code(root="experiments", name="experiments")

    # start training
    try:
        await trainer.train(
            config=train_config,
            train_dataset=ds_train.to_list(),
            val_dataset=ds_val.to_list(),
        )
    finally:
        trainer.close()


if __name__ == "__main__":
    try:
        prepare_environment()
        setup_logging(logging.INFO)
        args = parse_args()
        asyncio.run(main(args))
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logging.info("Training interrupted by user.")
        sys.exit(0)
