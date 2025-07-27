import asyncio
import random

from src.utils.env import prepare_environment
from src.models import load_art_model
from src.vllm_client import VllmClient, ArtVLLMClient
from src.trainer import TrainingConfig, PromptConfig, StopCriteria
from src.configs.prompts import DaCSystemPrompt
from easy2hard.trainer import Easy2HardTrainer

import art
from art.local import LocalBackend
from datasets import Dataset, load_dataset, DatasetDict


async def main():
    """
    Main function to run the training process.
    """
    prepare_environment("../api_keys")

    backend = LocalBackend(in_process=True)
    print("Local backend initialized")

    model_name = "unsloth/Qwen2.5-14B-Instruct"
    project_name = "easy2hard_dac_v2"

    # load model
    model, dir_config = await load_art_model(
        model_name=model_name,
        project_name=project_name,
        backend=backend,
        model_config=None,
    )

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

    vllm_server_ports = []
    for port in vllm_server_ports:
        vllm_client = VllmClient(port=port, base_model=model_name)
        inference_clients.append(vllm_client)

    # training configuration
    train_config = TrainingConfig(
        epochs=10,
        num_groups=2,
        group_size=10,
        eval_steps=None,
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
        dir_config=dir_config,
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
    asyncio.run(main())
    # Note: Ensure that the vLLM servers are running on the specified ports before starting the training process.
    # You can start the vLLM servers using the `run_vllm.server.py` script with the appropriate model configurations.
