from typing import Any

import chess

from src.trajectory import Trajectory
from src.agents import BaseAgent
from src.running.stage import RolloutStage
from src.running.rollout import RolloutTask

from experiments._framework.rewards import format_reward, behavior_reward
from experiments.chess.format import format_prompt
from experiments.chess.rewards import compute_reward, parse_move
from experiments.chess.chess_engine import EngineConfig, ChessConfig, MoveEvaluator


class ChessTask(RolloutTask):
    def __init__(self, configs: dict[str, Any]) -> None:
        super().__init__(configs)
        self.engine_config = EngineConfig.model_validate(configs["engine_config"])
        self.chess_config = ChessConfig.model_validate(configs["chess_config"])

    def format_prompt(self, sample: dict[str, Any]) -> str:
        return format_prompt(sample)

    async def score_trajectory(
        self,
        sample: dict[str, Any],
        trajectory: Trajectory,
        stage: RolloutStage,
        agent: BaseAgent,
    ) -> Trajectory:
        agent_answer = agent.parse_answer(trajectory.messages_and_responses[-1])
        evaluator = MoveEvaluator.for_process(self.engine_config)
        chess_config = self.chess_config.for_validation() if stage != RolloutStage.TRAIN else self.chess_config

        fen = sample["fen"]
        board = chess.Board(fen)

        move = parse_move(board, agent_answer or "<NO_ANSWER>")
        result = await evaluator.score(board, move, config=chess_config)

        answer_reward = compute_reward(result, config=chess_config)
        fmt_reward = format_reward(agent, trajectory)
        bhv_reward = behavior_reward(agent, trajectory)
        trajectory.reward = answer_reward + fmt_reward + bhv_reward

        trajectory.metrics.update(
            {
                "reward": trajectory.reward,
                "answer_reward": answer_reward,
                "format_reward": fmt_reward,
                "behavior_reward": bhv_reward,
                "parse_success": result.parse_success,
            }
        )

        trajectory.metadata.update(
            {
                "fen": fen,
                "ply": sample.get("ply"),
                "cp": result.cp,
                "agent_move": result.agent_move,
            }
        )

        return trajectory

    async def aclose(self):
        """
        Cleans process-local resources, including the chess engine.
        Should be called only when all rollouts on the process are finished and the engine is no longer needed.
        """
        MoveEvaluator.close_process()
