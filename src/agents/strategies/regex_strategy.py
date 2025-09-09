from __future__ import annotations
from typing import Any
import re

from src.agents.strategies.base import ParseStrategy, AgentTurn, TurnAction
from src.agents.exceptions import ParseError, InvalidActionError, StrategyError
from src.configs import DecompConfig
from src.utils.logging import create_logger

logger = create_logger(__name__)


class RegexParseStrategy(ParseStrategy):
    """Regex-based parsing strategy using guided generation."""
    
    def __init__(self):
        super().__init__()
        self._compiled_regex = None
        self._model_pattern = None
        
    def get_allowed_actions(self, decomp_config: DecompConfig, current_depth: int) -> list[TurnAction]:
        """Determine allowed actions based on decomposition rules."""
        # Special handling for leaf nodes in regex strategy
        if current_depth >= decomp_config.max_depth:
            decomp_config.max_rounds = 1  # Force only one round at leaf nodes
            
        is_leaf = current_depth >= decomp_config.max_depth
        has_rounds = decomp_config.total_rounds < decomp_config.max_rounds
        tasks_available = decomp_config.total_tasks < decomp_config.max_tasks

        allowed = [TurnAction.ANSWER]
        if has_rounds:
            allowed.append(TurnAction.THINK)
            if (not is_leaf) and tasks_available:
                allowed.append(TurnAction.ISSUE_TASK)

        return allowed
    
    def prepare_call_kwargs(self, allowed_actions: list[TurnAction], **kwargs) -> dict[str, Any]:
        """Prepare regex pattern for guided generation."""
        if not allowed_actions:
            raise StrategyError("At least one allowed action must be provided")
            
        # Build regex patterns
        alt = "|".join(re.escape(act.value) for act in allowed_actions)
        self._model_pattern = rf"^\s?Action: (?:{alt})\r?\nText: [\s\S]*$"
        parse_pattern = rf"^\s?Action: (?P<action>{alt})\r?\nText: (?P<text>[\s\S]*)$"
        self._compiled_regex = re.compile(parse_pattern)
        
        # Update kwargs for guided regex
        extra_body: dict = kwargs.setdefault("extra_body", {})
        extra_body.setdefault("include_stop_str_in_output", True)
        extra_body["guided_regex"] = self._model_pattern
        
        return kwargs
    
    def parse_response(self, content: str, allowed_actions: list[TurnAction]) -> AgentTurn:
        """Parse regex response into AgentTurn."""
        if not isinstance(content, str):
            raise ParseError("Content to parse must be a string", content, "Regex")

        if self._compiled_regex is None:
            # Rebuild regex if needed
            alt = "|".join(re.escape(act.value) for act in allowed_actions)
            parse_pattern = rf"^\s?Action: (?P<action>{alt})\r?\nText: (?P<text>[\s\S]*)$"
            self._compiled_regex = re.compile(parse_pattern)
            
        match = self._compiled_regex.match(content)
        if not match:
            logger.debug(f"Failed to match content against regex: {self._compiled_regex.pattern}")
            logger.debug(f"Raw content was: {content}")
            raise ParseError("Content does not match the required pattern", content, "Regex")

        action_val = match.group("action").strip()
        text_val = match.group("text").strip()

        try:
            action = TurnAction(action_val)
        except ValueError:
            raise ParseError(f"Invalid action '{action_val}'. Must be one of: {[a.value for a in TurnAction]}", content, "Regex")
            
        if action not in allowed_actions:
            raise InvalidActionError(action.value, [a.value for a in allowed_actions], content=content, strategy="Regex")

        return AgentTurn(action=action, text=text_val, raw=content)
    
    def parse_final_answer(self, content: str) -> str:
        """Parse final answer from regex response."""
        try:
            turn = self.parse_response(content, [TurnAction.ANSWER])
            return turn.text
        except Exception as e:
            logger.error(f"Failed to parse final answer: {e}")
            return content