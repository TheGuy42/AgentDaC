from typing import Any
import random

from omegaconf import OmegaConf

from src.trajectory import Trajectory
from src.agents import BaseAgent, NativePersistentAgent
from src.trainer import RolloutStage, VerlTrainer
from src.custom import VerlClient
from src.configs import DecompConfig

from experiments.chess_perst.format import format_prompt
from experiments.chess_perst.rewards import compute_reward
from experiments.chess_perst.chess_engine import EngineConfig, MoveEvaluator


class ChessNativeTrainer(VerlTrainer):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.engine_config = EngineConfig.model_validate(OmegaConf.to_container(self.config.custom_configs.engine_config, resolve=True))

    def create_agent(self, client: VerlClient, stage: RolloutStage) -> BaseAgent:
        max_depth = self.decomp_config.max_depth
        max_tasks = self.decomp_config.max_tasks
        max_rounds = self.decomp_config.max_rounds

        if stage == RolloutStage.TRAIN:
            if self.extra_config.get("randomize_decomp_depth", False):
                max_depth = random.randint(0, self.decomp_config.max_depth)
            if self.extra_config.get("randomize_decomp_tasks", False):
                max_tasks = random.randint(0, self.decomp_config.max_tasks)
            if self.extra_config.get("randomize_decomp_rounds", False):
                max_rounds = random.randint(0, self.decomp_config.max_rounds)

        decomp_config = DecompConfig(
            max_depth=max_depth,
            max_tasks=max_tasks,
            max_rounds=max_rounds,
        )

        return NativePersistentAgent(
            client=client,
            prompt_config=self.prompt_config,
            decomp_config=decomp_config,
            additional_histories=self.extra_config.get("additional_histories", False),
        )

    def format_prompt(self, sample: dict[str, Any]) -> str:
        return format_prompt(sample)

    async def score_trajectory(
        self,
        sample: dict,
        trajectory: Trajectory,
        stage: RolloutStage,
    ) -> Trajectory:
        ans_message = trajectory.messages()[-1]
        agent_answer = NativePersistentAgent.parse_answer(ans_message)

        evaluator = MoveEvaluator.for_process(self.engine_config)
        result = await evaluator.score(sample["fen"], agent_answer)
        reward = compute_reward(result, self.engine_config)
        trajectory.reward = reward

        trajectory.metrics.update(
            {
                "reward": reward,
                "parse_success": result.parse_success,
            }
        )

        trajectory.metadata.update(
            {
                "fen": sample["fen"],
                "ply": sample.get("ply"),
                "cp": result.cp,
                "agent_move": result.agent_move,
            }
        )

        return trajectory
