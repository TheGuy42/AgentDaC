from src.dac_agent import AgentNode
from src.trainer import Trainer, RolloutStage
from src.openai_types import UserMessage
from src.utils.markers import Markers
from src.utils.text import extract_answer, extract_between

from experiments.general_rewards import format_reward, behavior_reward
from experiments.saturn.rewards import answer_reward
from experiments.saturn.format import format_prompt

import art


class SaturnTrainer(Trainer):
    """
    See: https://arxiv.org/abs/2505.16368
    """
    
    def create_agent(self, stage: RolloutStage) -> AgentNode:
        client = self.vllm_router.next()
        return AgentNode(
            model_name=self.model.get_inference_name(),
            openai_client=client.openai_client,
            prompt_config=self.prompt_config,
            decomp_config=self.decomp_config,
        )

    async def forward_step(
        self,
        agent: AgentNode,
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
        ans_content = ans_message.get("content")
        assert ans_message["role"] == "assistant", f"Expected role 'assistant', got '{ans_message['role']}'"
        assert isinstance(ans_content, str), f"Expected content to be a string, got {type(ans_content)}"

        # Compute rewards
        trajectory.reward = 0.0
        ans_reward, parse_success = answer_reward(sample, ans_message)
        ans_reward = 3.0 * ans_reward
        trajectory.reward += ans_reward
        fmt_reward = format_reward(trajectory)
        trajectory.reward += fmt_reward
        bhv_reward = behavior_reward(trajectory)
        trajectory.reward += bhv_reward

        answer = sample["solution"].strip()
        agent_answer = extract_answer(ans_content)
        num_answers = len(extract_between(ans_content, Markers.ANSWER_START, Markers.ANSWER_END))

        # Update metrics
        trajectory.metrics.update(
            {
                "answer_reward": ans_reward,
                "format_reward": fmt_reward,
                "behavior_reward": bhv_reward,
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
                "clause": sample["clause"],
                "n_sat": sample["n_sat"],
                "k": sample["k"],
                "length": sample["length"],
            }
        )

        return trajectory
