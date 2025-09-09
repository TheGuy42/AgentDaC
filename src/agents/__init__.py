from src.agents.base import BaseAgent
from src.agents.marker_agent.marker_agent import MarkerAgent
from src.agents.json_agent.json_agent import JsonAgent
from src.agents.regex_agent.regex_agent import RegexAgent

# New unified implementations
from src.agents.conversation_agent import ConversationAgent
from src.agents.unified_agents import UnifiedJsonAgent, UnifiedRegexAgent, UnifiedMarkerAgent
from src.agents.factory import AgentFactory, AgentType

__all__ = [
    "BaseAgent",
    # Legacy agents (maintained for compatibility)
    "MarkerAgent",
    "JsonAgent",
    "RegexAgent",
    # New unified agents
    "ConversationAgent",
    "UnifiedJsonAgent",
    "UnifiedRegexAgent", 
    "UnifiedMarkerAgent",
    # Factory for creating agents
    "AgentFactory",
    "AgentType",
]
