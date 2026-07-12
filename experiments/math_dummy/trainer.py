from typing import Any

from src.agents import BaseAgent, DummyAgent
from src.inference import VerlClient
from src.trainer import RolloutStage, VerlTrainer
from src.trajectory import Trajectory

from experiments.math.format import format_prompt
from experiments.math.rewards import answer_reward


class MathDummyTrainer(VerlTrainer):
    def create_agent(self, client: VerlClient, stage: RolloutStage) -> BaseAgent:
        return DummyAgent(
            client=client,
            prompt_config=self.prompt_config,
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
        ans_message = trajectory.messages_and_responses[-1]
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
                "subject": sample["subject"],
                "level": sample["level"],
                "unique_id": sample["unique_id"],
            }
        )

        return trajectory
