import torch
import numpy as np
import os
import sys
from dotenv import load_dotenv
import random
import re
from datasets import Dataset, load_dataset
import regex

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import art
from art import Trajectory
from art.dev.model import InternalModelConfig
from art.local import LocalBackend
import openai
from openai import AsyncOpenAI

from training import Trainer
from sys_prompt import SystemPrompt, DaCSystemPrompt
from dac_agent import DACAgent, extract_text_between_markers
from vllm_client import VllmClient, VllmRouter


class Easy2HardTrainer(Trainer):
    def __init__(
        self,
        model_name: str,
        model_config: InternalModelConfig,
        project_name: str = "easy2hard-dac_agent",
        run_name: str | None = None,
        backend: LocalBackend = LocalBackend(path="./.art"),
        WANDB_API_KEY: str = "",
        OPENPIPE_API_KEY: str = "",
        seed: int = 42,
        gpu: int = 0,
        vllm_server_ports: list[int] = []
    ):
        super().__init__(
            model_name=model_name,
            model_config=model_config,
            project_name=project_name,
            run_name=run_name,
            backend=backend,
            WANDB_API_KEY=WANDB_API_KEY,
            OPENPIPE_API_KEY=OPENPIPE_API_KEY,
            seed=seed,
            gpu=gpu,
            vllm_server_ports=vllm_server_ports
        )

        print(f"Easy2HardTrainer:: Loading the dataset..")
        self.easy2hard_dataset = load_dataset("furonghuang-lab/Easy2Hard-Bench", "E2H-AMC")

        # The dataset is usually split into 'train' and 'test'
        self.train_data:Dataset = self.easy2hard_dataset['train']
        self.test_data:Dataset = self.easy2hard_dataset['eval']
        
        

    async def _train(self, epoch, n_rollouts, n_groups, vllm_router):
        # Split the dataset into n_groups
        # Each group will have n_rollouts samples
        # each group will have a greater difficulty level
        epoch_data_groups:list[Dataset] = []
        epoch_data = self.train_data.shuffle(seed=42)
        epoch_data = epoch_data.take(n_rollouts*n_groups)  # Take the first n_rollouts samples for training
        epoch_data = epoch_data.sort("item_difficulty", reverse=False)
        for i in range(n_groups):
            epoch_data_groups.append(epoch_data.shard(num_shards=n_groups, index=i, contiguous=True))

        train_groups = await art.gather_trajectory_groups(
            (
                art.TrajectoryGroup(
                    rollout(
                        sample=sample, 
                        vllm_client=vllm_router.__next__(),
                        model_config=self.model_config,
                    ) for i, sample in enumerate(epoch_data.iter(batch_size=1)) # Number of rollouts per group
                ) for epoch_data in epoch_data_groups # Number of groups to gather
            ),
            pbar_desc="gather",
        )

        await self.model.delete_checkpoints()
        await self.model.train(train_groups, config=art.TrainConfig(learning_rate=1e-5))


# @art.retry(exceptions=(openai.LengthFinishReasonError,))
async def rollout(
    sample,
    vllm_client: VllmClient,
    model_config: InternalModelConfig = None,
    
) -> art.Trajectory:
    question = sample['problem'][0]
    answer = sample['answer'][0]

    agent = DACAgent(
        client=vllm_client.client,
        model=vllm_client.get_inference_name(),
        # model_system_message=SystemPrompt.Qwen,
        # model_system_message="/no_think", #TODO: only relevant for Qwen3, remove if not using Qwen3
        dac_sys_prompt=DaCSystemPrompt.dac_sys_prompt_v2_3,
        leaf_sys_prompt=DaCSystemPrompt.dac_sys_prompt_v2_3_leaf,
        # dac_sys_prompt=prompt,
        max_depth=1,  # Set the maximum depth for the agent
        max_length=4,  # Limit the number of messages in a single chat
    )

    prompt = "Please answer the following question, write the final answer in the format <answer> final answer </answer>."
    message = {
        "role": "user",
        "content": f"{prompt} \n\"{question}\"",
    }
    try:
        max_tokens = model_config['max_seq_length'] if model_config and 'max_seq_length' in model_config else None
        trajectory = await agent.chat(message, max_tokens=max_tokens)
    except Exception as e:
        print("caught exception generating chat completion")
        print(e)
        # global failing_trajectory
        # failing_trajectory = trajectory
        return Trajectory(messages_and_choices=[message],reward=0)
        return e

    content = trajectory.messages()[-1]["content"]
    agent_answer = extract_text_between_markers(content, "<answer>", "</answer>")
    # answer = extract_boxed_content(answer)[-1]
    if len(agent_answer) == 0:
        trajectory.metrics["answer_given"] = 0
        trajectory.reward -= 3 # Penalize for no answer
        agent_answer = ""
    else:
        trajectory.metrics["answer_given"] = 1
        agent_answer = agent_answer[-1].strip()  # Get the last answer
        if agent_answer == answer:
            trajectory.reward += 1.5  # Reward for correct answer
            trajectory.metrics["correct_answer"] = 1
        else:
            trajectory.reward -= 1  # Penalize for incorrect answer
            trajectory.metrics["correct_answer"] = 0
    
    # Add the answer and agent_answer to the trajectory metrics
    trajectory.metadata["answer"] = answer
    trajectory.metadata["agent_answer"] = agent_answer
    trajectory.metadata['item_difficulty'] = sample['item_difficulty']
    trajectory.metadata['content'] = sample['content']

    return trajectory


def extract_boxed_content(text:str) -> list[str]:
    # Pattern explanation (same as before, but the DOTALL flag makes '.' match newlines):
    # \/boxed\{           - Matches the literal '/boxed{'
    # (                    - Start capturing group 1 (this is what we want to extract)
    #   (?:                - Start non-capturing group (for alternation)
    #     [^{}]            - Match any character that is NOT '{' or '}'
    #     |                - OR
    #     \{ (?R) \}       - Recursively match '{' followed by the entire pattern (including outer '/boxed{...}'), followed by '}'
    #   )* - Match the non-capturing group zero or more times (allowing empty content)
    # )                    - End capturing group 1
    # \}                   - Matches the literal '}'
    #
    # regex.DOTALL flag: Makes '.' match newlines, allowing the content to span multiple lines.
    # regex.MULTILINE flag: Not strictly necessary for this pattern, but good for patterns with ^ and $ anchors.
    # regex.IGNORECASE flag: Not needed here.
    pattern = r"\/boxed\{((?:[^{}]|(?R))*)\}"
    
    # Use regex.findall with the DOTALL flag
    matches = regex.findall(pattern, text, regex.DOTALL)
    return matches







