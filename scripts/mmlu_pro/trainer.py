from src.dac_agent import AgentNode
from src.dac_agent_single import SingleAgentNode
from src.trainer import Trainer
from src.dac_agent import ChatMessage
from src.utils.text import extract_answer

from scripts.mmlu_pro.rewards import answer_reward, format_reward, verify
from scripts.mmlu_pro.format import format_prompt

import art
from openai.types.chat.chat_completion import Choice


class MmluProTrainer(Trainer):
    def create_agent(self) -> AgentNode:
        client = self.get_client()
        return SingleAgentNode(
            model_name=client.get_inference_name(),
            openai_client=client.client,
            prompt_config=self.prompt_config,
            stop_criteria=self.stop_criteria.clone(),
        )

    async def forward_step(self, sample: dict, **kwargs) -> art.Trajectory:
        content = format_prompt(sample)
        agent = self.create_agent()
        message = ChatMessage(role="user", content=content)
        trajectory = await agent.chat(message, **kwargs)
        return trajectory

    async def rollout_step(self, sample: dict) -> art.Trajectory:
        # Perform a forward step to get the trajectory
        trajectory = await self.forward_step(sample)
        ans_message = ChatMessage.model_validate(trajectory.messages()[-1], from_attributes=True)

        # Update rewards
        ans_reward = answer_reward(sample, ans_message)
        trajectory.reward += ans_reward
        trajectory.metrics["answer_reward"] = ans_reward

        trajectory.metrics["format_reward"] = 0.0
        for item in trajectory.messages_and_choices:
            if isinstance(item, Choice):
                msg = ChatMessage.model_validate(item.message, from_attributes=True)
                fmt_reward = format_reward(msg)
                trajectory.reward += fmt_reward
                trajectory.metrics["format_reward"] += fmt_reward

        # Update metadata and metrics
        problem = format_prompt(sample)
        answer = sample["answer"].strip()
        agent_answer = extract_answer(ans_message.content)

        trajectory.metadata["problem"] = problem
        trajectory.metadata["category"] = sample["category"]
        trajectory.metadata["answer"] = answer
        trajectory.metadata["agent_answer"] = agent_answer

        is_correct = 1 if verify(answer, agent_answer) else 0
        trajectory.metrics["is_correct"] = is_correct
        return trajectory
