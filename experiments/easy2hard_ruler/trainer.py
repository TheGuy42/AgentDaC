from src.trainer import RolloutStage
from src.utils.markers import Markers
from src.utils.text import extract_answer, extract_between

from experiments.general_rewards import format_reward, behavior_reward
from experiments.easy2hard.trainer import Easy2HardTrainer
from experiments.easy2hard.rewards import answer_reward

import art
from art.rewards import ruler_score_group


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
        ans_reward, parse_success = answer_reward(sample, ans_message)
        ans_reward = 2.0 * ans_reward
        trajectory.reward += ans_reward
        fmt_reward = format_reward(trajectory)
        trajectory.reward += fmt_reward
        bhv_reward = behavior_reward(trajectory)
        trajectory.reward += bhv_reward

        answer = sample["answer"].strip()
        agent_answer = extract_answer(ans_content)
        num_answers = len(extract_between(ans_content, Markers.ANS_START, Markers.ANS_END))

        # Update metrics
        trajectory.metrics.update(
            {
                "reward_answer": ans_reward,
                "reward_format": fmt_reward,
                "reward_behavior": bhv_reward,
                "is_correct": ans_reward > 0.0,
                "gave_answer": num_answers > 0,
                "parse_success": parse_success,
            }
        )

        # Update metadata
        trajectory.metadata.update(
            {
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
    ) -> art.TrajectoryGroup | None:
        if stage != RolloutStage.Train:
            return group

        ruler_config = self.train_config().ruler_config
        if ruler_config is None:
            raise ValueError("Ruler config is not set in the training configuration.")

        if ruler_config.judge_model is None:
            raise ValueError("Ruler judge model is not set in the ruler configuration.")

        scored_group = await ruler_score_group(group, **ruler_config.model_dump(exclude_none=True, exclude_unset=True))
        if scored_group is None:
            return None

        for tr in scored_group.trajectories:
            tr.reward = 0.0
            tr.reward += tr.metrics["reward_answer"]
            tr.reward += tr.metrics["ruler_score"]

        return scored_group
