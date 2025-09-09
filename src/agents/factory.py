"""
Agent factory for creating agents with different parsing strategies.
Provides a clean API for agent creation and extensibility.
"""
from __future__ import annotations
from enum import Enum
from typing import Type, Any

from src.agents.base import BaseAgent
from src.agents.conversation_agent import ConversationAgent
from src.agents.unified_agents import UnifiedJsonAgent, UnifiedRegexAgent, UnifiedMarkerAgent
from src.agents.strategies import ParseStrategy, JsonParseStrategy, RegexParseStrategy, MarkerParseStrategy


class AgentType(str, Enum):
    """Enumeration of available agent types."""
    JSON = "json"
    REGEX = "regex" 
    MARKER = "marker"
    
    def __str__(self) -> str:
        return self.value


class AgentFactory:
    """Factory for creating agents with different parsing strategies."""
    
    # Mapping of agent types to their implementations
    _AGENT_REGISTRY: dict[AgentType, Type[BaseAgent]] = {
        AgentType.JSON: UnifiedJsonAgent,
        AgentType.REGEX: UnifiedRegexAgent,
        AgentType.MARKER: UnifiedMarkerAgent,
    }
    
    # Mapping of agent types to their parsing strategies
    _STRATEGY_REGISTRY: dict[AgentType, Type[ParseStrategy]] = {
        AgentType.JSON: JsonParseStrategy,
        AgentType.REGEX: RegexParseStrategy,
        AgentType.MARKER: MarkerParseStrategy,
    }
    
    @classmethod
    def create_agent(
        cls,
        agent_type: AgentType | str,
        openai_client,
        model_name: str,
        prompt_config,
        decomp_config,
        current_depth: int = 0,
        additional_histories: bool = False,
        **kwargs
    ) -> BaseAgent:
        """
        Create an agent of the specified type.
        
        Args:
            agent_type: Type of agent to create
            openai_client: OpenAI client for API calls
            model_name: Model name to use
            prompt_config: Prompt configuration
            decomp_config: Decomposition configuration
            current_depth: Current recursion depth
            additional_histories: Whether to store additional histories
            **kwargs: Additional arguments passed to agent constructor
            
        Returns:
            Configured agent instance
            
        Raises:
            ValueError: If agent_type is not supported
        """
        if isinstance(agent_type, str):
            try:
                agent_type = AgentType(agent_type.lower())
            except ValueError:
                supported = [t.value for t in AgentType]
                raise ValueError(f"Unsupported agent type '{agent_type}'. Supported: {supported}")
        
        if agent_type not in cls._AGENT_REGISTRY:
            supported = [t.value for t in AgentType]
            raise ValueError(f"Unsupported agent type '{agent_type}'. Supported: {supported}")
        
        agent_class = cls._AGENT_REGISTRY[agent_type]
        
        return agent_class(
            openai_client=openai_client,
            model_name=model_name,
            prompt_config=prompt_config,
            decomp_config=decomp_config,
            current_depth=current_depth,
            additional_histories=additional_histories,
            **kwargs
        )
    
    @classmethod
    def create_custom_agent(
        cls,
        parse_strategy: ParseStrategy,
        openai_client,
        model_name: str,
        prompt_config,
        decomp_config,
        current_depth: int = 0,
        additional_histories: bool = False,
        agent_class: Type[BaseAgent] | None = None,
        **kwargs
    ) -> BaseAgent:
        """
        Create an agent with a custom parsing strategy.
        
        Args:
            parse_strategy: Custom parsing strategy to use
            openai_client: OpenAI client for API calls
            model_name: Model name to use
            prompt_config: Prompt configuration
            decomp_config: Decomposition configuration
            current_depth: Current recursion depth
            additional_histories: Whether to store additional histories
            agent_class: Custom agent class (defaults to ConversationAgent)
            **kwargs: Additional arguments passed to agent constructor
            
        Returns:
            Configured agent instance with custom strategy
        """
        if agent_class is None:
            agent_class = ConversationAgent
            
        return agent_class(
            openai_client=openai_client,
            model_name=model_name,
            prompt_config=prompt_config,
            decomp_config=decomp_config,
            parse_strategy=parse_strategy,
            current_depth=current_depth,
            additional_histories=additional_histories,
            agent_class=agent_class,
            **kwargs
        )
    
    @classmethod
    def get_available_types(cls) -> list[str]:
        """Get list of available agent types."""
        return [agent_type.value for agent_type in AgentType]
    
    @classmethod
    def register_agent_type(
        cls, 
        agent_type: AgentType | str, 
        agent_class: Type[BaseAgent],
        strategy_class: Type[ParseStrategy] | None = None
    ) -> None:
        """
        Register a new agent type with the factory.
        
        Args:
            agent_type: Type identifier for the new agent
            agent_class: Agent class implementation
            strategy_class: Optional strategy class for the agent type
        """
        if isinstance(agent_type, str):
            # Create a new enum value dynamically
            agent_type = AgentType(agent_type.lower())
        
        cls._AGENT_REGISTRY[agent_type] = agent_class
        if strategy_class is not None:
            cls._STRATEGY_REGISTRY[agent_type] = strategy_class