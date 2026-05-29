from src.agents import BaseAgent, MarkerAgent
from src.trainer import ArtTrainer, RolloutStage
from src.aliases import UserMessage
from src.agents.marker_agent.markers import Markers, extract_between

from experiments.general_rewards import format_reward, behavior_reward
from experiments.mmlu_pro.rewards import answer_reward
from experiments.mmlu_pro.format import format_prompt

import art


class MmluProTrainer(ArtTrainer):
    def create_agent(self, stage: RolloutStage) -> BaseAgent:
        client = self.vllm_router.next()
        return MarkerAgent(
            model_name=client.get_inference_name(),
            openai_client=client.openai_client,
            prompt_config=self.prompt_config,
            decomp_config=self.decomp_config,
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
        return trajectory.to_art()

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

        answer = sample["answer"].strip()
        agent_answer = MarkerAgent.parse_answer(ans_message)
        num_answers = len(extract_between(ans_content, Markers.ANS_START, Markers.ANS_END))

        # Compute rewards
        trajectory.reward = 0.0
        ans_reward, parse_success = answer_reward(sample, agent_answer)
        ans_reward = 3.0 * ans_reward
        trajectory.reward += ans_reward
        fmt_reward = format_reward(trajectory)
        trajectory.reward += fmt_reward
        bhv_reward = 0.0 * behavior_reward(trajectory)
        trajectory.reward += bhv_reward

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
                "category": sample["category"],
            }
        )

        return trajectory
