from __future__ import annotations
from enum import StrEnum
from typing import Callable, Any
from dataclasses import dataclass

from src.agents import (
    BaseAgent,
    DummyAgent,
    MarkerAgent,
    JsonAgent,
    RegexAgent,
    PersistentAgent,
    NativePersistentAgent,
    ToolStatelessAgent,
    ToolPersistentAgent,
    ToolSubmitAgent,
)

from src.agents.tool_agent.parsing import build_tool_parser
from src.configs import DecompConfig, PromptConfig
from src.inference import InferenceClient


class AgentKey(StrEnum):
    MARKER = "marker"
    DUMMY = "dummy"
    JSON = "json"
    REGEX = "regex"
    PERST = "perst"
    NATIVE_PERST = "native_perst"
    TOOL_STATELESS = "tool_stateless"
    TOOL_PERSISTENT = "tool_persistent"
    TOOL_SUBMIT = "tool_submit"


REGISTRY: dict[str, Callable[..., BaseAgent]] = {}


@dataclass(frozen=True)
class AgentContext:
    agent_key: str
    prompt_config: PromptConfig
    decomp_config: DecompConfig
    extra_config: dict[str, Any]
    tool_parser: str | None = None
    reasoning_parser: str | None = None
    tokenizer: Any = None
    """A tokenizer object (verl passes its own) or a model name (ART); `build_tool_parser` takes either."""


def register_agent(key: str) -> Callable[..., Any]:
    """Register a new agent builder function under the given key."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        if key in REGISTRY:
            raise ValueError(f"{key!r} is already registered")

        REGISTRY[key] = func
        return func

    return decorator


def create_agent(client: InferenceClient, ctx: AgentContext) -> BaseAgent:
    """Build the agent named by `ctx.agent_key`."""
    if ctx.agent_key not in REGISTRY:
        raise ValueError(f"Agent key {ctx.agent_key!r} is not registered; expected one of {sorted(str(k) for k in REGISTRY)}.")
    return REGISTRY[ctx.agent_key](client=client, ctx=ctx)


@register_agent(AgentKey.DUMMY)
def _build_dummy(client: InferenceClient, ctx: AgentContext) -> BaseAgent:
    return DummyAgent(
        client=client,
        prompt_config=ctx.prompt_config,
    )


@register_agent(AgentKey.MARKER)
def _build_marker(client: InferenceClient, ctx: AgentContext) -> BaseAgent:
    return MarkerAgent(
        client=client,
        prompt_config=ctx.prompt_config,
        decomp_config=ctx.decomp_config,
        additional_histories=ctx.extra_config.get("additional_histories", False),
        strict=ctx.extra_config.get("marker_strict", True),
    )


@register_agent(AgentKey.JSON)
def _build_json(client: InferenceClient, ctx: AgentContext) -> BaseAgent:
    return JsonAgent(
        client=client,
        prompt_config=ctx.prompt_config,
        decomp_config=ctx.decomp_config,
        additional_histories=ctx.extra_config.get("additional_histories", False),
    )


@register_agent(AgentKey.REGEX)
def _build_regex(client: InferenceClient, ctx: AgentContext) -> BaseAgent:
    return RegexAgent(
        client=client,
        prompt_config=ctx.prompt_config,
        decomp_config=ctx.decomp_config,
        additional_histories=ctx.extra_config.get("additional_histories", False),
    )


@register_agent(AgentKey.PERST)
def _build_perst(client: InferenceClient, ctx: AgentContext) -> BaseAgent:
    return PersistentAgent(
        client=client,
        prompt_config=ctx.prompt_config,
        decomp_config=ctx.decomp_config,
        additional_histories=ctx.extra_config.get("additional_histories", False),
        force_thinking=ctx.extra_config.get("force_thinking", False),
    )


@register_agent(AgentKey.NATIVE_PERST)
def _build_native_perst(client: InferenceClient, ctx: AgentContext) -> BaseAgent:
    return NativePersistentAgent(
        client=client,
        prompt_config=ctx.prompt_config,
        decomp_config=ctx.decomp_config,
        additional_histories=ctx.extra_config.get("additional_histories", False),
    )


@register_agent(AgentKey.TOOL_STATELESS)
def _build_tool_stateless(client: InferenceClient, ctx: AgentContext) -> BaseAgent:
    if ctx.tool_parser is None:
        raise ValueError(f"Agent {ctx.agent_key!r} requires `tool_parser` to be set on its AgentContext")

    return ToolStatelessAgent(
        client=client,
        prompt_config=ctx.prompt_config,
        decomp_config=ctx.decomp_config,
        tool_parser=build_tool_parser(
            tool_parser=ctx.tool_parser,
            tokenizer=ctx.tokenizer,
            reasoning_parser=ctx.reasoning_parser,
        ),
        additional_histories=ctx.extra_config.get("additional_histories", False),
    )


@register_agent(AgentKey.TOOL_PERSISTENT)
def _build_tool_persistent(client: InferenceClient, ctx: AgentContext) -> BaseAgent:
    if ctx.tool_parser is None:
        raise ValueError(f"Agent {ctx.agent_key!r} requires `tool_parser` to be set on its AgentContext")

    return ToolPersistentAgent(
        client=client,
        prompt_config=ctx.prompt_config,
        decomp_config=ctx.decomp_config,
        tool_parser=build_tool_parser(
            tool_parser=ctx.tool_parser,
            tokenizer=ctx.tokenizer,
            reasoning_parser=ctx.reasoning_parser,
        ),
        additional_histories=ctx.extra_config.get("additional_histories", False),
    )


@register_agent(AgentKey.TOOL_SUBMIT)
def _build_tool_submit(client: InferenceClient, ctx: AgentContext) -> BaseAgent:
    if ctx.tool_parser is None:
        raise ValueError(f"Agent {ctx.agent_key!r} requires `tool_parser` to be set on its AgentContext")

    return ToolSubmitAgent(
        client=client,
        prompt_config=ctx.prompt_config,
        decomp_config=ctx.decomp_config,
        tool_parser=build_tool_parser(
            tool_parser=ctx.tool_parser,
            tokenizer=ctx.tokenizer,
            reasoning_parser=ctx.reasoning_parser,
        ),
        additional_histories=ctx.extra_config.get("additional_histories", False),
    )
