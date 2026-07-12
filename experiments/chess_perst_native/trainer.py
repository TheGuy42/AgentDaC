from src.agents import BaseAgent, NativePersistentAgent
from src.trainer import RolloutStage
from src.inference import VerlClient

from experiments.chess_perst.trainer import ChessTrainer


class ChessNativeTrainer(ChessTrainer):
    """Chess trainer using the native-reasoning persistent agent. Shares scoring,
    prompt formatting, and decomp-config logic with `ChessTrainer`; only the agent differs."""

    def create_agent(self, client: VerlClient, stage: RolloutStage) -> BaseAgent:
        return NativePersistentAgent(
            client=client,
            prompt_config=self.prompt_config,
            decomp_config=self._decomp_config_for_stage(stage),
            additional_histories=self.extra_config.get("additional_histories", False),
        )
