import sys
import torch
import pathlib
import asyncio
import argparse
import logging

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
from src.configs.art_model_config import available_configs
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
        "--train_config",
        type=str,
        help="Path to a JSON file containing training configuration parameters.",
    )

    parser.add_argument(
        "--prompt_config",
        type=str,
        help="Path to a JSON file containing prompt configuration parameters.",
    )

    parser.add_argument(
        "--stop_criteria",
        type=str,
        help="Path to a JSON file containing stop criteria parameters.",
    )

    return parser.parse_args()


def load_data() -> tuple[Dataset, Dataset]:
    """
    Load the Easy2Hard dataset.
    """
    data: Dataset = load_dataset(
        path="TIGER-Lab/MMLU-Pro",
        split="test",
    )  # type: ignore

    split_dict = data.train_test_split(test_size=0.3, seed=0)
    train_data = split_dict["train"]
    test_data = split_dict["test"]

    return train_data, test_data


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

    # load model
    model = await load_art_model(
        path_config=path_config,
        internal_config=None,
        openai_config=None,
        print_full=True,
    )

    # load dataset
    train_data, test_data = load_data()

    # create inference clients
    inference_clients: list[VllmClient] = [ArtVLLMClient(model)]
    for port in args.vllm_ports:
        vllm_client = VllmClient(port=port, base_model=path_config.model_name)
        inference_clients.append(vllm_client)

    # experiment setup
    if args.train_config:
        with open(args.train_config, "r") as f:
            train_config = TrainingConfig.model_validate_json(f.read())
    else:
        train_config = TrainingConfig(
            epochs=10,
            num_groups=2,
            group_size=10,
            log_every=1,
            eval_every=2,
            eval_size=250,
            art_config=art.types.TrainConfig(learning_rate=1e-5),
        )

    if args.prompt_config:
        with open(args.prompt_config, "r") as f:
            prompt_config = PromptConfig.model_validate_json(f.read())
    else:
        prompt_config = PromptConfig(
            system_root="dac_sys_prompt_gilad_root",
            system_inter="dac_sys_prompt_gilad_inter",
            system_leaf="dac_sys_prompt_gilad_leaf",
            tasks_depleted="tasks_depleted",
        )

    if args.stop_criteria:
        with open(args.stop_criteria, "r") as f:
            stop_criteria = StopCriteria.model_validate_json(f.read())
    else:
        stop_criteria = StopCriteria(
            max_depth=1,
            max_tasks=5,
            max_rounds=5,
        )

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
        await trainer.close()


if __name__ == "__main__":
    setup_logging(logging.INFO)
    args = parse_args()
    asyncio.run(main(args))
