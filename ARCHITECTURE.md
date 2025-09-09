# AgentDaC Unified Architecture

This document describes the new unified agent architecture that eliminates code duplication and improves extensibility.

## Overview

The original codebase had three separate agent implementations (JsonAgent, RegexAgent, MarkerAgent) with nearly identical logic (~200 lines each), resulting in ~600 lines of duplicate code. The new architecture eliminates this duplication while maintaining backward compatibility.

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

- **`BaseAgent`**: Abstract base class (unchanged for compatibility)
- **`ConversationAgent`**: Unified conversation logic  
- **`ParseStrategy`**: Abstract parsing strategy interface
- **`AgentFactory`**: Factory for creating agents
- **`AgentTurn`**: Common data structure for parsed responses

### Strategy Implementations

- **`JsonParseStrategy`**: JSON schema-based parsing
- **`RegexParseStrategy`**: Regex pattern-based parsing  
- **`MarkerParseStrategy`**: XML-like marker parsing

### Unified Agents

- **`UnifiedJsonAgent`**: JSON agent using new architecture
- **`UnifiedRegexAgent`**: Regex agent using new architecture
- **`UnifiedMarkerAgent`**: Marker agent using new architecture

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

The original agent classes are still available and fully functional:

```python
from src.agents import JsonAgent, RegexAgent, MarkerAgent

# Original APIs still work
json_agent = JsonAgent(openai_client, model_name, prompt_config, decomp_config)
```

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

## Migration Guide

### For New Code
Use the new `AgentFactory` and unified agents:

```python
# Old way
agent = JsonAgent(client, model, prompt_config, decomp_config)

# New way  
agent = AgentFactory.create_agent(AgentType.JSON, client, model, prompt_config, decomp_config)
```

### For Existing Code
No changes required - original classes still work:

```python
# This continues to work unchanged
agent = JsonAgent(client, model, prompt_config, decomp_config)
result = await agent.chat(message)
```

### Gradual Migration
You can migrate gradually by switching imports:

```python
# Change this:
from src.agents import JsonAgent

# To this:
from src.agents import UnifiedJsonAgent as JsonAgent
```

## Performance Impact

- **No performance regression** - same underlying logic
- **Reduced memory usage** - shared code paths
- **Better error handling** - custom exceptions provide more context

The new architecture maintains full API compatibility while providing significant improvements in code quality, extensibility, and maintainability.