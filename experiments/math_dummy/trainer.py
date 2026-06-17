from src.trajectory import Trajectory
from src.agents import BaseAgent, DummyAgent
from src.trainer import AglTrainer, RolloutStage

from experiments.math.format import format_prompt
from experiments.math.rewards import answer_reward

from openai import AsyncOpenAI
from typing import Any


class MathDummyTrainer(AglTrainer):
    def create_agent(self, client: AsyncOpenAI, model: str, stage: RolloutStage) -> BaseAgent:

        return DummyAgent(
            model_name=model,
            openai_client=client,
            prompt_config=self.prompt_config,
        )
        
    def format_prompt(self, sample: dict[str, Any]) -> str:
        return format_prompt(sample)

    async def score_trajectory(
        self,
        sample: dict,
        trajectory: Trajectory,
        stage: RolloutStage,
    ) -> Trajectory:
        ans_message = trajectory.messages()[-1]
        agent_answer = DummyAgent.parse_answer(ans_message)

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
