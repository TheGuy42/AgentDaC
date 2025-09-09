# AgentDaC Unified Architecture

This document describes the new unified agent architecture that eliminates code duplication and improves extensibility.

## Overview

The original codebase had three separate agent implementations (JsonAgent, RegexAgent, MarkerAgent) with nearly identical logic (~200 lines each), resulting in ~600 lines of duplicate code. The new architecture eliminates this duplication through a clean, unified design.

## Key Improvements

### 1. Strategy Pattern for Parsing

All parsing logic is now centralized into pluggable strategies:

```python
from src.agents.strategies import JsonParseStrategy, RegexParseStrategy, MarkerParseStrategy

# Each strategy handles its own parsing logic
strategy = JsonParseStrategy()
agent_turn = strategy.parse_response(content, allowed_actions)
```

### 2. Unified Conversation Logic

The `ConversationAgent` class contains all common conversation logic, eliminating duplication:

```python
from src.agents import ConversationAgent
from src.agents.strategies import JsonParseStrategy

# Create agent with any strategy
agent = ConversationAgent(
    openai_client=client,
    model_name="gpt-4", 
    prompt_config=config,
    decomp_config=decomp,
    parse_strategy=JsonParseStrategy()
)
```

### 3. Factory Pattern for Clean Creation

The `AgentFactory` provides a clean API for creating agents:

```python
from src.agents import AgentFactory, AgentType

# Simple agent creation
agent = AgentFactory.create_agent(
    agent_type=AgentType.JSON,
    openai_client=client,
    model_name="gpt-4",
    # ... other params
)

# Custom strategy
custom_agent = AgentFactory.create_custom_agent(
    parse_strategy=MyCustomStrategy(),
    # ... other params  
)
```

### 4. Improved Error Handling

Custom exception hierarchy provides better debugging:

```python
from src.agents.exceptions import ParseError, InvalidActionError, StrategyError

try:
    agent_turn = strategy.parse_response(content, actions)
except ParseError as e:
    print(f"Parse failed: {e}")
    print(f"Content: {e.content}")
    print(f"Strategy: {e.strategy}")
```

## Architecture Components

### Core Classes

- **`BaseAgent`**: Abstract base class
- **`ConversationAgent`**: Unified conversation logic  
- **`ParseStrategy`**: Abstract parsing strategy interface
- **`AgentFactory`**: Factory for creating agents
- **`AgentTurn`**: Common data structure for parsed responses

### Strategy Implementations

- **`JsonParseStrategy`**: JSON schema-based parsing
- **`RegexParseStrategy`**: Regex pattern-based parsing  
- **`MarkerParseStrategy`**: XML-like marker parsing

## Usage Examples

### Basic Usage

```python
from src.agents import AgentFactory, AgentType

# Create any agent type
agent = AgentFactory.create_agent(
    agent_type="json",  # or AgentType.JSON
    openai_client=openai_client,
    model_name="gpt-4",
    prompt_config=prompt_config,
    decomp_config=decomp_config
)

# Use the agent
response = await agent.answer("What is 2+2?")
```

### Custom Strategy

```python
from src.agents.strategies import ParseStrategy, AgentTurn, TurnAction

class MyCustomStrategy(ParseStrategy):
    def get_allowed_actions(self, decomp_config, current_depth):
        return [TurnAction.THINK, TurnAction.ANSWER]
    
    def prepare_call_kwargs(self, allowed_actions, **kwargs):
        # Customize API call parameters
        return kwargs
    
    def parse_response(self, content, allowed_actions):
        # Custom parsing logic
        return AgentTurn(action=TurnAction.ANSWER, text=content, raw=content)
    
    def parse_final_answer(self, content):
        return content

# Use custom strategy
agent = AgentFactory.create_custom_agent(
    parse_strategy=MyCustomStrategy(),
    # ... other params
)
```

### Extending the System

```python
# Register new agent type
AgentFactory.register_agent_type(
    agent_type="my_custom_type",
    agent_class=MyCustomAgent,
    strategy_class=MyCustomStrategy
)

# Create instance of new type  
agent = AgentFactory.create_agent("my_custom_type", ...)
```

## Backward Compatibility

## Benefits

### Code Reduction
- **~600 lines of duplicate code eliminated**
- **Single source of truth** for conversation logic
- **DRY principle** properly applied

### Extensibility 
- **New agent types**: 20 lines instead of 200+
- **Pluggable strategies** for different parsing approaches
- **Easy customization** without modifying core logic

### Maintainability
- **Centralized logic** easier to debug and modify
- **Consistent error handling** across all agents
- **Clear separation of concerns**

### Testability
- **Strategy testing** can be done in isolation
- **Conversation logic** tested once, works for all agents
- **Mocking simplified** with dependency injection

## Usage Guide

The new architecture provides a clean, unified API:

### Using AgentFactory (Recommended)

```python
from src.agents import AgentFactory, AgentType

# Create any agent type
agent = AgentFactory.create_agent(
    agent_type=AgentType.JSON,  # or "json", "regex", "marker"
    openai_client=client,
    model_name=model,
    prompt_config=prompt_config,
    decomp_config=decomp_config
)

result = await agent.chat(message)
```

### Using ConversationAgent Directly

```python
from src.agents import ConversationAgent
from src.agents.strategies import JsonParseStrategy

agent = ConversationAgent(
    openai_client=client,
    model_name=model,
    prompt_config=prompt_config,
    decomp_config=decomp_config,
    parse_strategy=JsonParseStrategy()
)
```

### Custom Parsing Strategies

```python
from src.agents import AgentFactory
from src.agents.strategies import ParseStrategy, AgentTurn, TurnAction

class CustomStrategy(ParseStrategy):
    def parse_response(self, content, allowed_actions):
        # Custom parsing logic
        return AgentTurn(action=TurnAction.ANSWER, text=content, raw=content)

# Use custom strategy
agent = AgentFactory.create_custom_agent(
    parse_strategy=CustomStrategy(),
    openai_client=client,
    model_name=model,
    prompt_config=prompt_config,
    decomp_config=decomp_config
)
```

## Architecture Benefits

- **Clean Design** - Single responsibility principle properly applied
- **Reduced Complexity** - Shared code paths eliminate duplication  
- **Better Error Handling** - Custom exceptions provide more context
- **High Extensibility** - New agent types require minimal code
- **Easy Testing** - Strategy testing can be done in isolation

The new architecture provides significant improvements in code quality, extensibility, and maintainability through its unified, strategy-based design.