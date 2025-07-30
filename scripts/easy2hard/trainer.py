from src.dac_agent import AgentNode
from src.dac_agent_single import SingleAgentNode
from src.trainer import Trainer
from src.dac_agent import ChatMessage
from src.utils.text import extract_answer
from scripts.easy2hard.rewards import answer_reward, format_reward, verify

import art
from openai.types.chat.chat_completion import Choice


class Easy2HardTrainer(Trainer):
    def create_agent(self) -> AgentNode:
        client = self.get_client()
        return SingleAgentNode(
            model_name=client.get_inference_name(),
            openai_client=client.client,
            prompt_config=self.prompt_config,
            stop_criteria=self.stop_criteria.clone(),
        )

    async def forward_step(self, sample: dict, **kwargs) -> art.Trajectory:
        instruction = (
            "The final answer should be written as valid LaTeX equation, starting with $ and ending with $. "
            "It should contain only the final result, without any additional text or explanation. "
            "Final answer format examples: $42$, $1,2,3,4$, $(1,2)$, $x^2$, $y=1$, $\\frac{1}{2}$, $\\sqrt{2} \\pi$, $\\text{Michael}$, $\\text{no}$, and so on."
        )

        problem = sample["problem"].strip()
        content = f"{problem}\n{instruction}"
        agent = self.create_agent()
        message = ChatMessage(role="user", content=content)
        trajectory = await agent.chat(message, **kwargs)
        return trajectory

    def split_into_groups(self, batch: list[dict], group_size: int) -> list[list[dict]]:
        # We split into groups according to difficulty
        batch = sorted(batch, key=lambda x: x["item_difficulty"])
        return [batch[i : i + group_size] for i in range(0, len(batch), group_size)]

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
        problem = sample["problem"].strip()
        answer = sample["answer"].strip()
        agent_answer = extract_answer(ans_message.content)

        trajectory.metadata["problem"] = problem
        trajectory.metadata["answer"] = answer
        trajectory.metadata["agent_answer"] = agent_answer
        trajectory.metadata["item_difficulty"] = sample["item_difficulty"]

        is_correct = 1 if verify(answer, agent_answer) else 0
        trajectory.metrics["is_correct"] = is_correct
        return trajectory
