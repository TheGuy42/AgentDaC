import sys
import torch
import pathlib
import asyncio
import random
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
from src.configs.prompts import DaCSystemPrompt
from src.configs.art_model_config import available_configs
from scripts.easy2hard.trainer import Easy2HardTrainer


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

    return parser.parse_args()


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
    model = await load_art_model(path_config=path_config, print_full=False)

    # load dataset
    dataset_dict: DatasetDict = load_dataset(
        path="furonghuang-lab/Easy2Hard-Bench",
        name="E2H-AMC",
        split=None,
    )  # type: ignore

    train_data: Dataset = dataset_dict["train"]
    test_data: Dataset = dataset_dict["eval"]

    # create inference clients
    inference_clients: list[VllmClient] = [ArtVLLMClient(model)]

    for port in args.vllm_ports:
        vllm_client = VllmClient(port=port, base_model=path_config.model_name)
        inference_clients.append(vllm_client)

    # training configuration
    train_config = TrainingConfig(
        epochs=10,
        num_groups=2,
        group_size=10,
        eval_every=None,
        verbose=False,
        min_reward_stdev=None,
        art_config=art.types.TrainConfig(learning_rate=1e-5),
        dev_art_config=None,
    )

    sys_prompt = PromptConfig(
        system_root=DaCSystemPrompt.dac_sys_prompt_gilad_root,
        system_inter=DaCSystemPrompt.dac_sys_prompt_gilad_inter,
        system_leaf=DaCSystemPrompt.dac_sys_prompt_gilad_leaf,
    )

    stop_criteria = StopCriteria(
        max_depth=1,
        max_tasks=5,
        max_rounds=5,
    )

    trainer = Easy2HardTrainer(
        model=model,
        client_list=inference_clients,
        path_config=path_config,
        train_config=train_config,
        prompt_config=sys_prompt,
        stop_criteria=stop_criteria,
    )

    # inference test
    idx = random.randint(0, len(train_data) - 1)
    print(f"Selected random index: {idx}")
    sample = train_data[idx]
    problem = sample["problem"]
    true_answer = sample["answer"]

    print(f"Problem: {problem}")
    print(f"Answer: {true_answer}")
    print("-" * 200)
    print()

    await trainer.predict([sample], verbose=True, seed=0)

    # train model
    await trainer.train(
        train_dataset=train_data.to_list(),
        eval_dataset=test_data.to_list(),
    )


if __name__ == "__main__":
    setup_logging(logging.INFO)
    args = parse_args()
    asyncio.run(main(args))
