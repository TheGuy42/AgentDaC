from typing import Any
import random
from openai import AsyncOpenAI

from src.trajectory import Trajectory
from src.agents import BaseAgent, PersistentAgent
from src.trainer import AglTrainer, RolloutStage
from src.configs import DecompConfig

from experiments.chess_perst.format import format_prompt
from experiments.chess_perst.chess_engine import EngineConfig, MoveEvaluator


class ChessTrainer(AglTrainer):
    """Train a PersistentAgent to pick the next chess move, scored by a local engine.

    Each runner subprocess lazily starts one engine via ``MoveEvaluator.for_process`` and
    reuses it across all of its rollouts. Engine and reward settings come from
    ``engine_config`` (loaded from ``engine_config.json`` in the config directory).
    """

    def __init__(self, *, engine_config: EngineConfig, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.engine_config = engine_config

    def create_agent(self, client: AsyncOpenAI, model: str, stage: RolloutStage) -> BaseAgent:
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

        return PersistentAgent(
            model_name=model,
            openai_client=client,
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
        agent_answer = PersistentAgent.parse_answer(ans_message)

        evaluator = MoveEvaluator.for_process(self.engine_config)
        score = await evaluator.score(sample["fen"], agent_answer)

        trajectory.reward = score.reward

        metrics = {"reward": score.reward, "parse_success": score.parse_success}
        if score.cp is not None:  # no resulting position to score for an illegal move
            metrics["cp"] = score.cp
        trajectory.metrics.update(metrics)

        trajectory.metadata.update(
            {
                "fen": sample["fen"],
                "ply": sample.get("ply"),
                "agent_move": score.agent_move,
            }
        )

        return trajectory
