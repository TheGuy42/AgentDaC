from src.trajectory import Trajectory
from src.agents import BaseAgent, MarkerAgent
from src.trainer import AglTrainer, RolloutStage
from src.aliases import UserMessage
from src.agents.marker_agent.markers import Markers, extract_between

from experiments.general_rewards import format_reward, behavior_reward
from experiments.big_code_bench.rewards import answer_reward, execute_code
from experiments.big_code_bench.format import format_prompt

from openai import AsyncOpenAI
from typing import Any


class BigCodeBenchTrainer(AglTrainer):
    def create_agent(self, client: AsyncOpenAI, model: str, stage: RolloutStage) -> BaseAgent:
        return MarkerAgent(
            openai_client=client,
            model_name=model,
            prompt_config=self.prompt_config,
            decomp_config=self.decomp_config,
        )

    def format_prompt(self, sample: dict[str, Any]) -> str:
        return format_prompt(sample)

    async def score_trajectory(
        self,
        sample: dict[str, Any],
        trajectory: Trajectory,
        stage: RolloutStage,
    ) -> Trajectory:
        ans_message = trajectory.messages()[-1]
        ans_content = ans_message.get("content")
        assert ans_message["role"] == "assistant", f"Expected role 'assistant', got '{ans_message['role']}'"
        assert isinstance(ans_content, str), f"Expected content to be a string, got {type(ans_content)}"

        answer = sample["canonical_solution"]  # sample["answer"].strip()
        agent_answer = MarkerAgent.parse_answer(ans_message)
        num_answers = len(extract_between(ans_content, Markers.ANS_START, Markers.ANS_END))

        result = execute_code(sample, agent_answer)

        # Compute rewards
        trajectory.reward = 0.0
        ans_reward = 3.0 * answer_reward(result)
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
                "is_correct": result.returncode,
                "gave_answer": num_answers > 0,
                "execution_time": result.execution_time if result.execution_time is not None else -1000000.0,
            }
        )

        # Update metadata
        trajectory.metadata.update(
            {
                "answer": answer,
                "agent_answer": agent_answer,
            }
        )

        return trajectory
