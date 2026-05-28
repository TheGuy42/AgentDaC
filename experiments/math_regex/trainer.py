from src.agents import BaseAgent, RegexAgent
from src.trainer import Trainer, RolloutStage
from src.openai_types import UserMessage
from src.configs import DecompConfig

from experiments.math.format import format_prompt
from experiments.math.rewards import answer_reward

import art
import random


class MathRegexTrainer(Trainer):
    def create_agent(self, stage: RolloutStage) -> BaseAgent:
        client = self.vllm_router.next()

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
            model_name=client.get_inference_name(),
            openai_client=client.openai_client,
            prompt_config=self.prompt_config,
            decomp_config=decomp_config,
            additional_histories=self.extra_config.get("additional_histories", False),
        )

    async def forward_step(
        self,
        agent: BaseAgent,
        sample: dict,
        stage: RolloutStage,
    ) -> art.Trajectory:
        content = format_prompt(sample)
        message = UserMessage(role="user", content=content)
        kwargs = self.rollout_config.get_kwargs(stage)
        trajectory = await agent.chat(message, **kwargs)
        return trajectory

    async def score_trajectory(
        self,
        sample: dict,
        trajectory: art.Trajectory,
        stage: RolloutStage,
    ) -> art.Trajectory:
        ans_message = trajectory.messages()[-1]
        agent_answer = RegexAgent.parse_answer(ans_message)

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
