from src.trajectory import Trajectory
from src.agents import BaseAgent, JsonAgent
from src.trainer import RolloutStage, VerlTrainer
from src.custom import VerlClient
from src.configs import DecompConfig

from experiments.math.rewards import answer_reward
from experiments.math.format import format_prompt

import random
from typing import Any


class MathJsonTrainer(VerlTrainer):
    def create_agent(self, client: VerlClient, stage: RolloutStage) -> BaseAgent:
        max_depth = self.decomp_config.max_depth
        max_tasks = self.decomp_config.max_tasks
        max_rounds = self.decomp_config.max_rounds

        if stage == RolloutStage.TRAIN:
            if self.extra_config.get("randomize_decomp_depth", False):
                max_depth = random.randint(0, self.decomp_config.max_depth)
            if self.extra_config.get("randomize_decomp_tasks", False):
                max_tasks = random.randint(0, self.decomp_config.max_tasks)
            if self.extra_config.get("randomize_decomp_rounds", False):
                max_rounds = random.randint(0, self.decomp_config.max_rounds)

        decomp_config = DecompConfig(
            max_depth=max_depth,
            max_tasks=max_tasks,
            max_rounds=max_rounds,
        )

        return JsonAgent(
            client=client,
            prompt_config=self.prompt_config,
            decomp_config=decomp_config,
        )

    def format_prompt(self, sample: dict[str, Any]) -> str:
        return format_prompt(sample)

    async def score_trajectory(self, sample: dict[str, Any], trajectory: Trajectory, stage: RolloutStage) -> Trajectory:

        ans_message = trajectory.messages()[-1]
        agent_answer = JsonAgent.parse_answer(ans_message)

        # Compute rewards
        trajectory.reward = 0.0
        ans_reward, parse_success = answer_reward(sample, agent_answer)
        trajectory.reward += ans_reward

        # Update metrics
        trajectory.metrics.update(
            {
                "answer_reward": ans_reward,
                "is_correct": ans_reward > 0.0,
                "parse_success": parse_success,
            }
        )

        # Update metadata
        trajectory.metadata.update(
            {
                "answer": sample["answer"],
                "agent_answer": agent_answer,
                "subject": sample["subject"],
                "level": sample["level"],
                "unique_id": sample["unique_id"],
            }
        )

        return trajectory
