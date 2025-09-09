from __future__ import annotations
from typing import Any

from src.agents.strategies.base import ParseStrategy, AgentTurn, TurnAction
from src.agents.strategies.markers import Markers, extract_answer, extract_tasks
from src.configs import DecompConfig
from src.utils.logging import create_logger

logger = create_logger(__name__)


class MarkerParseStrategy(ParseStrategy):
    """Marker-based parsing strategy using XML-like tags."""
    
    def get_allowed_actions(self, decomp_config: DecompConfig, current_depth: int) -> list[TurnAction]:
        """
        Marker strategy doesn't use explicit action enumeration like JSON/Regex.
        Instead it uses the presence of markers to determine behavior.
        This method returns all possible actions for compatibility.
        """
        # For marker strategy, we allow all actions and determine behavior based on content
        return [TurnAction.THINK, TurnAction.ISSUE_TASK, TurnAction.ANSWER]
    
    def prepare_call_kwargs(self, allowed_actions: list[TurnAction], **kwargs) -> dict[str, Any]:
        """Prepare kwargs for marker-based parsing."""
        # Set up stop tokens for marker strategy
        extra_body = kwargs.setdefault("extra_body", {})
        extra_body.setdefault("include_stop_str_in_output", True)
        kwargs.setdefault("stop", [Markers.TASK_END, Markers.ANS_END])
        return kwargs
    
    def parse_response(self, content: str, allowed_actions: list[TurnAction]) -> AgentTurn:
        """
        Parse marker response. For marker strategy, we determine the action based on content.
        This is different from JSON/Regex which have explicit action fields.
        """
        if not isinstance(content, str):
            raise ValueError("Content to parse must be a string.")

        # Extract tasks and determine action based on content
        tasks = extract_tasks(content)
        
        if tasks:
            # If tasks are present, this is an ISSUE_TASK action
            # We'll use the first task as the text (marker agent can handle multiple tasks)
            return AgentTurn(
                action=TurnAction.ISSUE_TASK,
                text=content,  # Keep full content for marker parsing
                raw=content
            )
        else:
            # No tasks found, treat as answer
            answer_text = extract_answer(content)
            return AgentTurn(
                action=TurnAction.ANSWER,
                text=answer_text,
                raw=content
            )
    
    def parse_final_answer(self, content: str) -> str:
        """Parse final answer from marker response."""
        return extract_answer(content)
        
    def parse_tasks(self, message) -> list:
        """
        Helper method specific to marker strategy for parsing tasks.
        Extracts tasks from marker-formatted content.
        """
        if message["role"] != "assistant":
            raise ValueError("Message role must be 'assistant' to extract tasks.")

        content = message.get("content")
        if not isinstance(content, str):
            logger.error(f"Expected message content to be a string, got {type(content).__name__}")
            raise ValueError("Message content must be a string.")

        from src.openai_types import UserMessage
        tasks = extract_tasks(content)
        return [UserMessage(role="user", content=task) for task in tasks]