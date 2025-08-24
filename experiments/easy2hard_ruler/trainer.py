from src.trainer import RolloutStage
from src.utils.markers import Markers
from src.utils.text import extract_answer, extract_between

from experiments.general_rewards import format_reward, behavior_reward
from experiments.easy2hard.trainer import Easy2HardTrainer
from experiments.easy2hard.rewards import answer_reward, verify
from experiments.easy2hard.format import format_prompt

import art


class Easy2HardRulerTrainer(Easy2HardTrainer):
    async def score_trajectory(
        self,
        sample: dict,
        trajectory: art.Trajectory,
        stage: RolloutStage,
    ) -> art.Trajectory:
        ans_message = trajectory.messages()[-1]
        ans_content = ans_message.get("content")
        assert ans_message["role"] == "assistant", f"Expected role 'assistant', got '{ans_message['role']}'"
        assert isinstance(ans_content, str), f"Expected content to be a string, got {type(ans_content)}"

        # Compute rewards
        trajectory.reward = 0.0
        ans_reward = 2.0 * answer_reward(sample, ans_message)
        trajectory.reward += ans_reward
        fmt_reward = format_reward(trajectory)
        trajectory.reward += fmt_reward
        bhv_reward = behavior_reward(trajectory)
        trajectory.reward += bhv_reward

        problem = format_prompt(sample)
        answer = sample["answer"].strip()
        agent_answer = extract_answer(ans_content)
        num_answers = len(extract_between(ans_content, Markers.ANSWER_START, Markers.ANSWER_END))

        # Update metrics
        trajectory.metrics.update(
            {
                "reward_answer": ans_reward,
                "reward_format": fmt_reward,
                "reward_behavior": bhv_reward,
                "is_correct": int(verify(answer, agent_answer)),
                "gave_answer": int(num_answers > 0),
            }
        )

        # Update metadata
        trajectory.metadata.update(
            {
                "problem": problem,
                "answer": answer,
                "agent_answer": agent_answer,
                "item_difficulty": sample["item_difficulty"],
                "rating": sample["rating"],
                "rating_quantile": sample["rating_quantile"],
                "contest": sample["contest"],
            }
        )

        return trajectory

    async def score_group(
        self,
        group: art.TrajectoryGroup,
        stage: RolloutStage,
    ) -> art.TrajectoryGroup:
        for tr in group.trajectories:
            tr.reward = 0.0
            tr.reward += tr.metrics.get("reward_answer", 0.0)
            # tr.reward += tr.metrics.get("reward_format", 0.0)
            # tr.reward += tr.metrics.get("reward_behavior", 0.0)
            tr.reward += tr.metrics.get("ruler_score", 0.0)
        return group
