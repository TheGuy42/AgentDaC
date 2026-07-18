from __future__ import annotations

from src.aliases import ToolSchema
from src.agents.tool_agent.persistent_agent import ToolPersistentAgent, CREATE_NEW_SUB_AGENT
from src.agents.tool_agent.schemas import tool_schema


class ToolStatelessAgent(ToolPersistentAgent):
    # NOTE: The same as a persistent agent, 
    # but only with the ability of creating new sub-agents.

    def _build_tools(self) -> dict[str, ToolSchema]:
        if self.decomp_config.is_leaf(self.current_depth):
            return {}  # leaf: no delegation, answers directly
        return {
            CREATE_NEW_SUB_AGENT: tool_schema(
                name=CREATE_NEW_SUB_AGENT,
                desc=(
                    "Delegate a self-contained sub-task to a fresh sub-agent and receive its answer. "
                    "The sub-agent starts with NO memory or context of this conversation, so the `text` must "
                    "be fully self-contained: restate the complete position and everything needed to solve it."
                ),
                arg_name="text",
                arg_desc=(
                    "A clear, fully self-contained description of the sub-task, including the complete "
                    "position/board and the exact answer format required."
                ),
            )
        }

    def available_tools(self) -> list[ToolSchema]:
        DC = self.decomp_config
        available = []
        if (not DC.is_leaf(self.current_depth)) and DC.has_tasks():
            available.append(self.tool_map[CREATE_NEW_SUB_AGENT])
        return available

    def create_subagent(self) -> ToolStatelessAgent:
        agent = ToolStatelessAgent(
            client=self.client,
            prompt_config=self.prompt_config,
            decomp_config=self.decomp_config,
            tool_parser=self.tool_parser,
            current_depth=self.current_depth + 1,
            additional_histories=False,
            verbose=self.verbose,
        )

        if self.additional_histories:
            agent.trajectory.histories = self.trajectory.histories
            
        return agent
