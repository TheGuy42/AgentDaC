from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
from dataclasses import dataclass
from enum import Enum

from src.openai_types import Message
from src.configs import DecompConfig


class TurnAction(str, Enum):
    """Common turn actions available to all agents."""
    THINK = "think"
    ISSUE_TASK = "issue_task"
    ANSWER = "answer"
    ERROR = "error"
    
    def __str__(self) -> str:
        return self.value


@dataclass
class AgentTurn:
    """Represents a parsed agent turn with action and content."""
    action: TurnAction
    text: str
    raw: Any  # Original raw content for debugging


class ParseStrategy(ABC):
    """Abstract base class for different parsing strategies."""
    
    def __init__(self, *actions: TurnAction) -> None:
        self.actions = tuple(actions)
        
    @abstractmethod
    def get_allowed_actions(self, decomp_config: DecompConfig, current_depth: int) -> list[TurnAction]:
        """
        Determine allowed actions based on current state.
        
        Args:
            decomp_config: Current decomposition configuration
            current_depth: Current recursion depth
            
        Returns:
            List of allowed actions for this turn
        """
        pass
    
    @abstractmethod 
    def prepare_call_kwargs(self, allowed_actions: list[TurnAction], **kwargs) -> dict[str, Any]:
        """
        Prepare additional kwargs for the OpenAI API call based on parsing strategy.
        
        Args:
            allowed_actions: Actions allowed for this turn
            **kwargs: Additional arguments from caller
            
        Returns:
            Modified kwargs dict for API call
        """
        pass
    
    @abstractmethod
    def parse_response(self, content: str, allowed_actions: list[TurnAction]) -> AgentTurn:
        """
        Parse the model's response into an AgentTurn.
        
        Args:
            content: Raw content from the model
            allowed_actions: Actions that were allowed for this turn
            
        Returns:
            Parsed AgentTurn
            
        Raises:
            ValueError: If content cannot be parsed or action not allowed
        """
        pass
        
    @abstractmethod
    def parse_final_answer(self, content: str) -> str:
        """
        Parse the final answer from a response.
        
        Args:
            content: Raw content from the model
            
        Returns:
            Extracted answer text
        """
        pass