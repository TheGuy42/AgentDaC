"""
Custom exceptions for agent-related errors.
Provides better error handling and debugging capabilities.
"""


class AgentError(Exception):
    """Base exception for all agent-related errors."""
    pass


class ParseError(AgentError):
    """Raised when agent response parsing fails."""
    
    def __init__(self, message: str, content: str | None = None, strategy: str | None = None):
        super().__init__(message)
        self.content = content
        self.strategy = strategy
    
    def __str__(self) -> str:
        parts = [super().__str__()]
        if self.strategy:
            parts.append(f"Strategy: {self.strategy}")
        if self.content:
            parts.append(f"Content: {self.content[:200]}{'...' if len(self.content) > 200 else ''}")
        return "\n".join(parts)


class InvalidActionError(ParseError):
    """Raised when an action is not allowed in the current context."""
    
    def __init__(self, action: str, allowed_actions: list[str], **kwargs):
        message = f"Action '{action}' not allowed. Allowed actions: {allowed_actions}"
        super().__init__(message, **kwargs)
        self.action = action
        self.allowed_actions = allowed_actions


class ConfigurationError(AgentError):
    """Raised when agent configuration is invalid."""
    pass


class AgentStateError(AgentError):
    """Raised when agent is in an invalid state for the requested operation."""
    pass


class StrategyError(AgentError):
    """Raised when there's an error with the parsing strategy."""
    pass