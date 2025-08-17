from src.dac_agent_single import SingleAgentNode
from src.trainer import Trainer
from src.dac_agent import ChatMessage
from src.configs.markers import Markers
from src.utils.text import extract_answer, extract_between

from experiments.general_rewards import format_reward, behavior_reward
from experiments.easy2hard_ruler.rewards import answer_reward, verify
from experiments.easy2hard_ruler.format import format_prompt

import art


class Easy2HardRulerTrainer(Trainer):
    async def forward_step(self, sample: dict, **kwargs) -> art.Trajectory:
        client = self.vllm_router.next()
        agent = SingleAgentNode(
            model_name=self.model.get_inference_name(),
            openai_client=client.openai_client,
            prompt_config=self.prompt_config,
            stop_criteria=self.stop_criteria,
        )

        content = format_prompt(sample)
        message = ChatMessage(role="user", content=content)
        trajectory = await agent.chat(message, **kwargs)
        return trajectory

    async def score_trajectory(self, sample: dict, trajectory: art.Trajectory) -> art.Trajectory:
        ans_message = ChatMessage.model_validate(trajectory.messages()[-1], from_attributes=True)

        # Compute rewards
        trajectory.reward = 0.0
        ans_reward = answer_reward(sample, ans_message)
        trajectory.reward += ans_reward
        fmt_reward = format_reward(trajectory)
        trajectory.reward += fmt_reward
        bhv_reward = behavior_reward(trajectory)
        trajectory.reward += bhv_reward

        problem = format_prompt(sample)
        answer = sample["answer"].strip()
        agent_answer = extract_answer(ans_message.content)
        num_answers = len(extract_between(ans_message.content, Markers.ANSWER_START, Markers.ANSWER_END))

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

    async def score_group(self, group: art.TrajectoryGroup) -> art.TrajectoryGroup:
        for tr in group.trajectories:
            tr.reward = 0.0
            tr.reward += tr.metrics.get("reward_answer", 0.0)
            # tr.reward += tr.metrics.get("reward_format", 0.0)
            # tr.reward += tr.metrics.get("reward_behavior", 0.0)
            tr.reward += tr.metrics.get("ruler_score", 0.0)
        return group
