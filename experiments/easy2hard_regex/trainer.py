from src.trajectory import Trajectory
from src.agents import BaseAgent, RegexAgent
from src.trainer import RolloutStage, VerlTrainer
from src.inference import VerlClient
from src.aliases import UserMessage
from src.configs import DecompConfig

from experiments.easy2hard.rewards import answer_reward
from experiments.easy2hard.format import format_prompt

import random
from typing import Any

class Easy2HardRegexTrainer(VerlTrainer):
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

        return RegexAgent(
            client=client,
            prompt_config=self.prompt_config,
            decomp_config=decomp_config,
            additional_histories=self.extra_config.get("additional_histories", False),
        )

    def format_prompt(self, sample: dict[str, Any]) -> str:
        return format_prompt(sample)

    async def score_trajectory(
        self,
        sample: dict[str, Any],
        trajectory: Trajectory,
        stage: RolloutStage,
        agent: BaseAgent,
    ) -> Trajectory:
        ans_message = trajectory.messages()[-1]
        agent_answer = agent.parse_answer(ans_message)

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
                "item_difficulty": sample["item_difficulty"],
                "rating": sample["rating"],
                "rating_quantile": sample["rating_quantile"],
                "contest": sample["contest"],
            }
        )

        return trajectory
