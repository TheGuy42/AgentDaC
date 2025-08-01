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
from src.models import load_art_model, PathConfig
from src.vllm_client import VllmClient, ArtVLLMClient
from src.trainer import TrainingConfig, PromptConfig, StopCriteria
from src.configs.art_model_config import available_configs, ArtConfig
from experiments.easy2hard.trainer import Easy2HardTrainer


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
        "--project_name",
        type=str,
        default="easy2hard_dac",
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
    dataset_dict: DatasetDict = load_dataset(
        path="furonghuang-lab/Easy2Hard-Bench",
        name="E2H-AMC",
        split=None,
    )  # type: ignore

    train_data: Dataset = dataset_dict["train"]
    test_data: Dataset = dataset_dict["eval"]
    return train_data, test_data


def load_configs(config_dir: str | pathlib.Path) -> dict[str, Any]:
    if isinstance(config_dir, str):
        config_dir = pathlib.Path(config_dir)

    if not config_dir.exists():
        raise ValueError(f"Config directory '{config_dir}' does not exist.")

    if not config_dir.is_dir():
        raise ValueError(f"Config directory '{config_dir}' is not a directory.")

    configs_dict = {}

    art_config_path = config_dir / "art_config.json"
    if art_config_path.exists():
        art_config = ArtConfig.model_validate_json(art_config_path.read_text())
        configs_dict["art_config"] = art_config

    train_config_path = config_dir / "train_config.json"
    if train_config_path.exists():
        train_config = TrainingConfig.model_validate_json(train_config_path.read_text())
        configs_dict["train_config"] = train_config

    prompt_config_path = config_dir / "prompt_config.json"
    if prompt_config_path.exists():
        prompt_config = PromptConfig.model_validate_json(prompt_config_path.read_text())
        configs_dict["prompt_config"] = prompt_config

    stop_criteria_path = config_dir / "stop_criteria.json"
    if stop_criteria_path.exists():
        stop_criteria = StopCriteria.model_validate_json(stop_criteria_path.read_text())
        configs_dict["stop_criteria"] = stop_criteria

    return configs_dict


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
    trainer = Easy2HardTrainer(
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
    setup_logging(logging.WARNING)
    args = parse_args()
    asyncio.run(main(args))
