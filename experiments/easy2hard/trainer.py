from src.dac_agent import AgentNode
from src.dac_agent_single import SingleAgentNode
from src.trainer import Trainer
from src.dac_agent import ChatMessage
from src.utils.text import extract_answer

from experiments.general_rewards import format_reward, behavior_reward
from experiments.easy2hard.rewards import answer_reward, verify

import art


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

        # Compute rewards
        trajectory.reward = 0.0
        ans_reward = answer_reward(sample, ans_message)
        trajectory.reward += ans_reward
        fmt_reward = format_reward(trajectory)
        trajectory.reward += fmt_reward
        bhv_reward = behavior_reward(trajectory)
        trajectory.reward += bhv_reward

        problem = sample["problem"].strip()
        answer = sample["answer"].strip()
        agent_answer = extract_answer(ans_message.content)

        # Update metrics
        trajectory.metrics.update({
            "answer_reward": ans_reward,
            "format_reward": fmt_reward,
            "behavior_reward": bhv_reward,
            "total_reward": trajectory.reward,
            "is_correct": int(verify(answer, agent_answer)),
        })

        # Update metadata
        trajectory.metadata.update({
            "problem": problem,
            "answer": answer,
            "agent_answer": agent_answer,
            "item_difficulty": sample["item_difficulty"],
        })

        return trajectory
