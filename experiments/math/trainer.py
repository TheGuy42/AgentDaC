from __future__ import annotations

from typing import Any

from src.agents.base import BaseAgent
from src.running.stage import RolloutStage
from src.trajectory import Trajectory

from experiments._framework.trainer import ExperimentTrainer
from experiments._framework.rewards import format_reward, behavior_reward
from experiments.math.format import format_prompt
from experiments.math.rewards import answer_reward


class MathTrainer(ExperimentTrainer):
    def format_prompt(self, sample: dict[str, Any]) -> str:
        return format_prompt(sample)

    async def score_trajectory(
        self,
        sample: dict[str, Any],
        trajectory: Trajectory,
        stage: RolloutStage,
        agent: BaseAgent,
    ) -> Trajectory:
        agent_answer = agent.parse_answer(trajectory.messages_and_responses[-1])

        scale = float(self.extra_config.get("answer_scale", 1.0))
        ans_raw, parse_success = answer_reward(sample, agent_answer or "<NO_ANSWER>")
        ans_reward = scale * ans_raw
        fmt_reward = format_reward(agent, trajectory)
        bhv_reward = behavior_reward(agent, trajectory)
        trajectory.reward = ans_reward + fmt_reward + bhv_reward

        trajectory.metrics.update(
            {
                "answer_reward": ans_reward,
                "format_reward": fmt_reward,
                "behavior_reward": bhv_reward,
                "is_correct": ans_reward > 0.0,
                "gave_answer": agent_answer is not None,
                "parse_success": parse_success,
            }
        )

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
