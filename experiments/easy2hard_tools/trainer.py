from src.dac_agent_tools import AgentToolNode
from src.trainer import Trainer, RolloutStage
from src.openai_types import UserMessage
from src.utils.markers import Markers
from src.utils.text import extract_answer, extract_between

from experiments.general_rewards import format_reward, behavior_reward
from experiments.easy2hard.rewards import answer_reward, verify
from experiments.easy2hard.format import format_prompt

import art


class Easy2HardToolTrainer(Trainer):
    def create_agent(self) -> AgentToolNode:
        client = self.vllm_router.next()
        return AgentToolNode(
            model_name=self.model.get_inference_name(),
            openai_client=client.openai_client,
            prompt_config=self.prompt_config,
            decomp_config=self.decomp_config,
        )

    async def forward_step(self, sample: dict, stage: RolloutStage) -> art.Trajectory:
        agent = self.create_agent()
        content = format_prompt(sample)
        message = UserMessage(role="user", content=content)
        kwargs = self.rollout_config.get_kwargs(stage)
        trajectory = await agent.chat(message, **kwargs)
        return trajectory

    async def predict_step(self, sample: dict, stage: RolloutStage) -> str:
        agent = self.create_agent()
        content = format_prompt(sample)
        message = UserMessage(role="user", content=content)
        kwargs = self.rollout_config.get_kwargs(stage)
        answer_message = await agent.answer(message, **kwargs)

        answer = answer_message.get("content")
        assert answer_message["role"] == "assistant", f"Expected role 'assistant', got '{answer_message['role']}'"
        assert isinstance(answer, str), f"Expected content to be a string, got {type(answer)}"
        return answer

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
                "answer_reward": ans_reward,
                "format_reward": fmt_reward,
                "behavior_reward": bhv_reward,
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
            tr.reward += tr.metrics.get("answer_reward", 0.0)
            # tr.reward += tr.metrics.get("format_reward", 0.0)
            # tr.reward += tr.metrics.get("behavior_reward", 0.0)
            tr.reward += tr.metrics.get("ruler_score", 0.0)
        return group
