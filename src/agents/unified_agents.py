"""
Unified agent implementations that use the ConversationAgent with different parsing strategies.
These replace the original JsonAgent, RegexAgent, and MarkerAgent with much simpler implementations.
"""
from __future__ import annotations

from src.agents.conversation_agent import ConversationAgent
from src.agents.strategies import JsonParseStrategy, RegexParseStrategy, MarkerParseStrategy
from src.openai_types import Message
from src.utils.logging import create_logger

logger = create_logger(__name__)


class UnifiedJsonAgent(ConversationAgent):
    """JSON agent implemented using the unified conversation agent."""
    
    def __init__(
        self,
        openai_client,
        model_name: str,
        prompt_config,
        decomp_config,
        current_depth: int = 0,
        additional_histories: bool = False,
    ):
        super().__init__(
            openai_client=openai_client,
            model_name=model_name,
            prompt_config=prompt_config,
            decomp_config=decomp_config,
            parse_strategy=JsonParseStrategy(),
            current_depth=current_depth,
            additional_histories=additional_histories,
            agent_class=UnifiedJsonAgent,
        )
    
    @staticmethod
    def parse_answer(message: Message) -> str:
        """Parse final answer from JSON response."""
        if message["role"] != "assistant":
            logger.error(f"Expected message role 'assistant', got {message['role']}")
            raise ValueError("Message role must be 'assistant' to extract answer.")

        content = message.get("content")
        if not isinstance(content, str):
            logger.error(f"Expected message content to be a string, got {type(content)}")
            raise ValueError("Message content must be a string.")

        try:
            strategy = JsonParseStrategy()
            return strategy.parse_final_answer(content)
        except Exception as e:
            logger.error(f"Failed to parse final answer: {e}")
            return content


class UnifiedRegexAgent(ConversationAgent):
    """Regex agent implemented using the unified conversation agent."""
    
    def __init__(
        self,
        openai_client,
        model_name: str,
        prompt_config,
        decomp_config,
        current_depth: int = 0,
        additional_histories: bool = False,
    ):
        super().__init__(
            openai_client=openai_client,
            model_name=model_name,
            prompt_config=prompt_config,
            decomp_config=decomp_config,
            parse_strategy=RegexParseStrategy(),
            current_depth=current_depth,
            additional_histories=additional_histories,
            agent_class=UnifiedRegexAgent,
        )
    
    @staticmethod
    def parse_answer(message: Message) -> str:
        """Parse final answer from regex response."""
        if message["role"] != "assistant":
            logger.error(f"Expected message role 'assistant', got {message['role']}")
            raise ValueError("Message role must be 'assistant' to extract answer.")

        content = message.get("content")
        if not isinstance(content, str):
            logger.error(f"Expected message content to be a string, got {type(content)}")
            raise ValueError("Message content must be a string.")

        try:
            strategy = RegexParseStrategy()
            return strategy.parse_final_answer(content)
        except Exception as e:
            logger.error(f"Failed to parse final answer: {e}")
            return content


class UnifiedMarkerAgent(ConversationAgent):
    """Marker agent implemented using the unified conversation agent."""
    
    def __init__(
        self,
        openai_client,
        model_name: str,
        prompt_config,
        decomp_config,
        current_depth: int = 0,
        additional_histories: bool = False,
    ):
        super().__init__(
            openai_client=openai_client,
            model_name=model_name,
            prompt_config=prompt_config,
            decomp_config=decomp_config,
            parse_strategy=MarkerParseStrategy(),
            current_depth=current_depth,
            additional_histories=additional_histories,
            agent_class=UnifiedMarkerAgent,
        )
    
    @staticmethod
    def parse_answer(message: Message) -> str:
        """Parse final answer from marker response."""
        if message["role"] != "assistant":
            logger.error(f"Expected message role 'assistant', got {message['role']}")
            raise ValueError("Message role must be 'assistant' to extract answer.")

        content = message.get("content")
        if not isinstance(content, str):
            logger.error(f"Expected message content to be a string, got {type(content)}")
            raise ValueError("Message content must be a string.")

        strategy = MarkerParseStrategy()
        return strategy.parse_final_answer(content)