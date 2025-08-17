from src.dac_agent import AgentNode
from src.dac_agent_single import SingleAgentNode
from src.trainer import Trainer
from src.dac_agent import ChatMessage
from src.configs.markers import Markers
from src.utils.text import extract_answer, extract_between

from experiments.general_rewards import format_reward, behavior_reward
from experiments.BigCodeBench.rewards import answer_reward
from experiments.BigCodeBench.format import format_prompt, create_test_code

import art

from experiments.BigCodeBench.Server.code_client import CodeClient, ExecutionResult


class BigCodeBenchTrainer(Trainer):
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
        problem = format_prompt(sample)
        answer = sample["canonical_solution"]  # sample["answer"].strip()
        agent_answer = extract_answer(ans_message.content)
        num_answers = len(extract_between(ans_message.content, Markers.ANSWER_START, Markers.ANSWER_END))

        agent_test_code = create_test_code(sample, agent_answer)
        client = CodeClient(port=8002, timeout_buffer=5)
        # Execute the agent's answer code
        result = client.execute_code(agent_test_code, execution_timeout=60)

        if not isinstance(self.model, art.TrainableModel):
            raise ValueError("Model must be an instance of TrainableModel for scoring.")

        train_step = await self.model.get_step()
        # Compute rewards
        trajectory.reward = 0.0
        ans_reward = answer_reward(ans_message, result) if train_step > 5 else 0.0
        trajectory.reward += ans_reward
        fmt_reward = format_reward(trajectory)
        trajectory.reward += fmt_reward
        bhv_reward = behavior_reward(trajectory)
        trajectory.reward += bhv_reward

        # Update metrics
        trajectory.metrics.update(
            {
                "answer_reward": ans_reward,
                "format_reward": fmt_reward,
                "behavior_reward": bhv_reward,
                "is_correct": int(result.returncode),
                "gave_answer": int(num_answers > 0),
                "execution_time": result.execution_time if result.execution_time is not None else -1000000.0,
            }
        )

        # Update metadata
        trajectory.metadata.update(
            {
                "problem": problem,
                "answer": answer,
                "agent_answer": agent_answer,
                # "item_difficulty": sample["item_difficulty"],
            }
        )

        return trajectory

    async def score_group(self, group: art.TrajectoryGroup) -> art.TrajectoryGroup:
        return group
