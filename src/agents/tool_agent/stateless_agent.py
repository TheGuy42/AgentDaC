from __future__ import annotations
from src.agents.tool_agent.persistent_agent import ToolPersistentAgent, ToolPersistentActions
from src.agents.tool_agent.schemas import GuidedTools
from src.configs import ToolSpecs


class ToolStatelessAgent(ToolPersistentAgent):
    """
    A stateless variant of :class:`ToolPersistentAgent`.

    Unlike its parent, this agent does not retain memory or context between
    calls. Any sub-agents it creates are also stateless
    """

    def _create_tools(self) -> GuidedTools:
        specs = self.prompt_config.tools
        if not isinstance(specs, ToolSpecs):
            raise ValueError(f"Expected ToolSpecs, got {type(specs).__name__}")
        
        if self.decomp_config.is_leaf(self.current_depth):
            return GuidedTools(specs)  # leaf: no delegation, answers directly
        return GuidedTools(specs, ToolPersistentActions.CREATE_NEW_SUB_AGENT)

    def allowed_actions(self) -> list[str]:
        DC = self.decomp_config
        if DC.is_leaf(self.current_depth) or not DC.has_tasks():
            return []
        return [ToolPersistentActions.CREATE_NEW_SUB_AGENT]

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
            self.trajectory.histories.append(agent.trajectory)

        return agent
