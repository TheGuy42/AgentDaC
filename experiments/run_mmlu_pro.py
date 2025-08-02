import sys
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
from src.vllm_client import VllmClient, ArtVLLMClient
from src.trainer import TrainingConfig, PromptConfig, StopCriteria
from src.configs.art_configs import available_configs, ArtConfig
from experiments.mmlu_pro.trainer import MmluProTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the MMLU-Pro experiment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help=f"The name or path of the model to serve. Available models are: {available_configs()}",
    )

    parser.add_argument(
        "--project_name",
        type=str,
        default="mmlu_pro_dac",
        help="The name of the project for saving results.",
    )

    parser.add_argument(
        "--run_name",
        type=str,
        default="",
        help="The name of the run for saving results.",
    )

    parser.add_argument(
        "--gpu",
        type=int,
        nargs="+",
        default=[0],
        help=f"The ID of the GPU(s) to use (e.g., 0 or '0,1'). Available GPUs: {list(range(torch.cuda.device_count()))}",
    )

    parser.add_argument(
        "--vllm_ports",
        type=int,
        nargs="*",
        default=[],
        help="List of endpoint ports for vLLM servers.",
    )

    parser.add_argument(
        "--config-dir",
        type=str,
        help="Directory containing experiment configuration files.",
    )

    return parser.parse_args()


def load_data() -> tuple[Dataset, Dataset]:
    data: Dataset = load_dataset(
        path="TIGER-Lab/MMLU-Pro",
        split="test",
    )  # type: ignore

    split_dict = data.train_test_split(test_size=0.3, seed=0)
    train_data = split_dict["train"]
    test_data = split_dict["test"]
    return train_data, test_data


def load_configs(config_dir: str | pathlib.Path) -> dict[str, Any]:
    if isinstance(config_dir, str):
        config_dir = pathlib.Path(config_dir)

    configs = {
        "art_config": load_base_model(ArtConfig, config_dir / "art_config.json", do_raise=False),
        "train_config": load_base_model(TrainingConfig, config_dir / "train_config.json", do_raise=False),
        "prompt_config": load_base_model(PromptConfig, config_dir / "prompt_config.json", do_raise=False),
        "stop_criteria": load_base_model(StopCriteria, config_dir / "stop_criteria.json", do_raise=False),
    }

    return {k: v for k, v in configs.items() if v is not None}


async def main(args: argparse.Namespace):
    """
    Main function to run the training process.
    """
    prepare_environment()

    path_config = PathConfig(
        model_name=args.model,
        project_name=args.project_name,
        run_name=args.run_name,
    )

    # Defaults
    art_config: ArtConfig | None = None

    train_config: TrainingConfig = TrainingConfig(
        epochs=10,
        num_groups=2,
        group_size=10,
        log_every=1,
        eval_every=2,
        eval_size=250,
        art_config=art.types.TrainConfig(learning_rate=1e-5),
    )

    prompt_config: PromptConfig = PromptConfig(
        system_root="dac_sys_prompt_gilad_root",
        system_inter="dac_sys_prompt_gilad_inter",
        system_leaf="dac_sys_prompt_gilad_leaf",
        tasks_depleted="tasks_depleted",
    )

    stop_criteria: StopCriteria = StopCriteria(
        max_depth=1,
        max_tasks=5,
        max_rounds=5,
    )

    # Load configurations if provided
    if args.config_dir:
        configs = load_configs(args.config_dir)
        art_config = configs.get("art_config", art_config)
        train_config = configs.get("train_config", train_config)
        prompt_config = configs.get("prompt_config", prompt_config)
        stop_criteria = configs.get("stop_criteria", stop_criteria)

    # load model
    model = await load_art_model(
        path_config=path_config,
        art_config=art_config,
        print_full=True,
    )

    # load dataset
    train_data, test_data = load_data()

    # create inference clients
    inference_clients: list[VllmClient] = [ArtVLLMClient(model)]
    for port in args.vllm_ports:
        vllm_client = VllmClient(port=port, base_model=path_config.model_name)
        inference_clients.append(vllm_client)

    # create and configure the trainer
    trainer = MmluProTrainer(
        model=model,
        inference_clients=inference_clients,
        path_config=path_config,
        train_config=train_config,
        prompt_config=prompt_config,
        stop_criteria=stop_criteria,
    )

    # start training
    try:
        await trainer.train(
            train_dataset=train_data.to_list(),
            eval_dataset=test_data.to_list(),
        )
    finally:
        trainer.close()


if __name__ == "__main__":
    setup_logging(logging.INFO)
    args = parse_args()
    asyncio.run(main(args))
