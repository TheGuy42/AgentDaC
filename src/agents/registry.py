from __future__ import annotations
from enum import StrEnum
from typing import TYPE_CHECKING, Callable, Any
from omegaconf import OmegaConf

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

from src.running.stage import RolloutStage
from src.inference import InferenceClient
from src.agents.tool_agent.parsing import build_tool_parser

if TYPE_CHECKING:
    from experiments._framework.trainer import ExperimentTrainer


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


REGISTRY: dict[str, Callable[[ExperimentTrainer, InferenceClient, RolloutStage], BaseAgent]] = {}


def register_agent(key: str) -> Callable[..., Any]:
    """Register a new agent builder function under the given key."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        if key in REGISTRY:
            raise ValueError(f"{key!r} is already registered")

        REGISTRY[key] = func
        return func

    return decorator


def create_agent(key: str, trainer: "ExperimentTrainer", client: InferenceClient, stage: RolloutStage) -> BaseAgent:
    if key not in REGISTRY:
        raise ValueError(f"Agent key {key} is not registered.")
    return REGISTRY[key](trainer, client, stage)


@register_agent(key=AgentKey.DUMMY)
def _build_dummy(trainer: "ExperimentTrainer", client: InferenceClient, stage: RolloutStage) -> BaseAgent:
    return DummyAgent(
        client=client,
        prompt_config=trainer.prompt_config,
    )


@register_agent(key=AgentKey.MARKER)
def _build_marker(trainer: "ExperimentTrainer", client: InferenceClient, stage: RolloutStage) -> BaseAgent:
    return MarkerAgent(
        client=client,
        prompt_config=trainer.prompt_config,
        decomp_config=trainer.build_decomp_config(stage),
        additional_histories=trainer.extra_config.get("additional_histories", False),
        strict=trainer.extra_config.get("marker_strict", True),
    )


@register_agent(key=AgentKey.JSON)
def _build_json(trainer: "ExperimentTrainer", client: InferenceClient, stage: RolloutStage) -> BaseAgent:
    return JsonAgent(
        client=client,
        prompt_config=trainer.prompt_config,
        decomp_config=trainer.build_decomp_config(stage),
        additional_histories=trainer.extra_config.get("additional_histories", False),
    )


@register_agent(key=AgentKey.REGEX)
def _build_regex(trainer: "ExperimentTrainer", client: InferenceClient, stage: RolloutStage) -> BaseAgent:
    return RegexAgent(
        client=client,
        prompt_config=trainer.prompt_config,
        decomp_config=trainer.build_decomp_config(stage),
        additional_histories=trainer.extra_config.get("additional_histories", False),
    )


@register_agent(key=AgentKey.PERST)
def _build_perst(trainer: "ExperimentTrainer", client: InferenceClient, stage: RolloutStage) -> BaseAgent:
    return PersistentAgent(
        client=client,
        prompt_config=trainer.prompt_config,
        decomp_config=trainer.build_decomp_config(stage),
        additional_histories=trainer.extra_config.get("additional_histories", False),
        force_thinking=trainer.extra_config.get("force_thinking", False),
    )


@register_agent(key=AgentKey.NATIVE_PERST)
def _build_native_perst(trainer: "ExperimentTrainer", client: InferenceClient, stage: RolloutStage) -> BaseAgent:
    return NativePersistentAgent(
        client=client,
        prompt_config=trainer.prompt_config,
        decomp_config=trainer.build_decomp_config(stage),
        additional_histories=trainer.extra_config.get("additional_histories", False),
    )


@register_agent(key=AgentKey.TOOL_STATELESS)
def _build_tool_stateless(trainer: "ExperimentTrainer", client: InferenceClient, stage: RolloutStage) -> BaseAgent:
    return ToolStatelessAgent(
        client=client,
        prompt_config=trainer.prompt_config,
        decomp_config=trainer.build_decomp_config(stage),
        tool_parser=build_tool_parser(
            tool_parser=trainer.config.actor_rollout_ref.rollout.multi_turn.format,
            reasoning_parser=OmegaConf.select(trainer.config, "actor_rollout_ref.rollout.engine_kwargs.vllm.reasoning_parser", default=None),
            tokenizer=trainer.tokenizer,  # type: ignore[arg-type]
        ),
        additional_histories=trainer.extra_config.get("additional_histories", False),
    )


@register_agent(key=AgentKey.TOOL_PERSISTENT)
def _build_tool_persistent(trainer: "ExperimentTrainer", client: InferenceClient, stage: RolloutStage) -> BaseAgent:
    return ToolPersistentAgent(
        client=client,
        prompt_config=trainer.prompt_config,
        decomp_config=trainer.build_decomp_config(stage),
        tool_parser=build_tool_parser(
            tool_parser=trainer.config.actor_rollout_ref.rollout.multi_turn.format,
            reasoning_parser=OmegaConf.select(trainer.config, "actor_rollout_ref.rollout.engine_kwargs.vllm.reasoning_parser", default=None),
            tokenizer=trainer.tokenizer,  # type: ignore[arg-type]
        ),
        additional_histories=trainer.extra_config.get("additional_histories", False),
    )


@register_agent(key=AgentKey.TOOL_SUBMIT)
def _build_tool_submit(trainer: "ExperimentTrainer", client: InferenceClient, stage: RolloutStage) -> BaseAgent:
    return ToolSubmitAgent(
        client=client,
        prompt_config=trainer.prompt_config,
        decomp_config=trainer.build_decomp_config(stage),
        tool_parser=build_tool_parser(
            tool_parser=trainer.config.actor_rollout_ref.rollout.multi_turn.format,
            reasoning_parser=OmegaConf.select(trainer.config, "actor_rollout_ref.rollout.engine_kwargs.vllm.reasoning_parser", default=None),
            tokenizer=trainer.tokenizer,  # type: ignore[arg-type]
        ),
        additional_histories=trainer.extra_config.get("additional_histories", False),
    )
