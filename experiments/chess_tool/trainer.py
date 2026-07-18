from omegaconf import OmegaConf

from src.agents import BaseAgent, ToolStatelessAgent, ToolPersistentAgent, ToolSubmitAgent
from src.trainer import RolloutStage
from src.inference import VerlClient
from src.agents.tool_agent.parsing import build_tool_parser
from experiments.chess_perst.trainer import ChessTrainer


# Native-tool-calling flows (see src/agents/tool_agent).
FLOWS: dict[str, type[ToolStatelessAgent] | type[ToolPersistentAgent]] = {
    "stateless": ToolStatelessAgent,
    "persistent": ToolPersistentAgent,
    "persistent_submit": ToolSubmitAgent,
}


class ChessToolTrainer(ChessTrainer):
    def create_agent(self, client: VerlClient, stage: RolloutStage) -> BaseAgent:
        flow = self.extra_config.get("flow", "stateless")
        agent_cls = FLOWS.get(flow)
        if agent_cls is None:
            raise ValueError(f"Unknown flow {flow!r}; expected one of {sorted(FLOWS)}.")

        tool_parser = build_tool_parser(
            name=self.config.actor_rollout_ref.rollout.multi_turn.format,
            reasoning_parser=OmegaConf.select(self.config, "actor_rollout_ref.rollout.engine_kwargs.vllm.reasoning_parser", default=None),
            tokenizer=self.tokenizer,  # type: ignore
        )

        return agent_cls(
            client=client,
            prompt_config=self.prompt_config,
            decomp_config=self._decomp_config_for_stage(stage),
            tool_parser=tool_parser,
            additional_histories=self.extra_config.get("additional_histories", False),
        )
