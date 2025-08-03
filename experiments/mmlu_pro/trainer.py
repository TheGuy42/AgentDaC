from src.dac_agent import AgentNode
from src.dac_agent_single import SingleAgentNode
from src.trainer import Trainer
from src.dac_agent import ChatMessage
from src.configs.markers import Markers
from src.utils.text import extract_answer, extract_between

from experiments.general_rewards import format_reward, behavior_reward
from experiments.mmlu_pro.rewards import answer_reward, verify

import art


class MmluProTrainer(Trainer):
    def create_agent(self) -> AgentNode:
        client = self.vllm_router.next()
        return SingleAgentNode(
            model_name=self.inference_name,
            openai_client=client.openai_client,
            prompt_config=self.prompt_config,
            stop_criteria=self.stop_criteria.clone(),
        )

    def format_prompt(self, sample: dict) -> str:
        question = sample["question"].strip()
        category = sample["category"].strip()
        options: list = sample["options"]
        letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

        instruction = (
            f"The following is a multiple choice question (with answers) about {category}. Only one answer is correct. "
            f"Answer with X where X is the correct letter choice. The final answer should contain only a letter choice."
        )

        fmt_options = "Options are:\n"
        for option, letter in zip(options, letters):
            fmt_options += f"({letter}): {option}" + "\n"

        fmt_question = f"Q: {question}\n{fmt_options}"
        content = f"{instruction}\n{fmt_question}".strip()
        return content

    async def forward_step(self, sample: dict, **kwargs) -> art.Trajectory:
        content = self.format_prompt(sample)
        agent = self.create_agent()
        message = ChatMessage(role="user", content=content)
        trajectory = await agent.chat(message, **kwargs)
        return trajectory

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

        problem = self.format_prompt(sample)
        answer = sample["answer"].strip()
        agent_answer = extract_answer(ans_message.content)
        num_answers = len(extract_between(ans_message.content, Markers.ANSWER_START, Markers.ANSWER_END))

        # Update metrics
        trajectory.metrics.update({
            "answer_reward": ans_reward,
            "format_reward": fmt_reward,
            "behavior_reward": bhv_reward,
            "total_reward": trajectory.reward,
            "is_correct": int(verify(answer, agent_answer)),
            "gave_answer": int(num_answers > 0),
        })

        # Update metadata
        trajectory.metadata.update({
            "problem": problem,
            "answer": answer,
            "agent_answer": agent_answer,
            "category": sample["category"]
        })

        return trajectory
