from __future__ import annotations

import functools
from dataclasses import dataclass
from enum import StrEnum
from typing import Callable, TYPE_CHECKING

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

from src.agents.tool_agent.parsing import build_tool_parser
from src.inference import InferenceClient
from src.running.stage import RolloutStage

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


@dataclass(frozen=True)
class AgentSpec:
    agent_cls: type[BaseAgent]
    build: Callable[["ExperimentTrainer", InferenceClient, RolloutStage], BaseAgent]


def _build_dummy(trainer: "ExperimentTrainer", client: InferenceClient, stage: RolloutStage) -> BaseAgent:
    return DummyAgent(
        client=client,
        prompt_config=trainer.prompt_config,
    )


def _build_marker(trainer: "ExperimentTrainer", client: InferenceClient, stage: RolloutStage) -> BaseAgent:
    return MarkerAgent(
        client=client,
        prompt_config=trainer.prompt_config,
        decomp_config=trainer.build_decomp_config(stage),
        additional_histories=trainer.extra_config.get("additional_histories", False),
        strict=trainer.extra_config.get("marker_strict", True),
    )


def _build_json(trainer: "ExperimentTrainer", client: InferenceClient, stage: RolloutStage) -> BaseAgent:
    return JsonAgent(
        client=client,
        prompt_config=trainer.prompt_config,
        decomp_config=trainer.build_decomp_config(stage),
        additional_histories=trainer.extra_config.get("additional_histories", False),
    )


def _build_regex(trainer: "ExperimentTrainer", client: InferenceClient, stage: RolloutStage) -> BaseAgent:
    return RegexAgent(
        client=client,
        prompt_config=trainer.prompt_config,
        decomp_config=trainer.build_decomp_config(stage),
        additional_histories=trainer.extra_config.get("additional_histories", False),
    )


def _build_perst(trainer: "ExperimentTrainer", client: InferenceClient, stage: RolloutStage) -> BaseAgent:
    return PersistentAgent(
        client=client,
        prompt_config=trainer.prompt_config,
        decomp_config=trainer.build_decomp_config(stage),
        additional_histories=trainer.extra_config.get("additional_histories", False),
        force_thinking=trainer.extra_config.get("force_thinking", False),
    )


def _build_native_perst(trainer: "ExperimentTrainer", client: InferenceClient, stage: RolloutStage) -> BaseAgent:
    return NativePersistentAgent(
        client=client,
        prompt_config=trainer.prompt_config,
        decomp_config=trainer.build_decomp_config(stage),
        additional_histories=trainer.extra_config.get("additional_histories", False),
    )


def _build_tool_stateless(trainer: "ExperimentTrainer", client: InferenceClient, stage: RolloutStage) -> BaseAgent:
    return ToolStatelessAgent(
        client=client,
        prompt_config=trainer.prompt_config,
        decomp_config=trainer.build_decomp_config(stage),
        tool_parser=build_native_parser(
            tool_parser=trainer.config.actor_rollout_ref.rollout.multi_turn.format,
            reasoning_parser=OmegaConf.select(trainer.config, "actor_rollout_ref.rollout.engine_kwargs.vllm.reasoning_parser", default=None),
            tokenizer=trainer.tokenizer,  # type: ignore[arg-type]
        ),
        additional_histories=trainer.extra_config.get("additional_histories", False),
    )


def _build_tool_persistent(trainer: "ExperimentTrainer", client: InferenceClient, stage: RolloutStage) -> BaseAgent:
    return ToolPersistentAgent(
        client=client,
        prompt_config=trainer.prompt_config,
        decomp_config=trainer.build_decomp_config(stage),
        tool_parser=build_native_parser(
            tool_parser=trainer.config.actor_rollout_ref.rollout.multi_turn.format,
            reasoning_parser=OmegaConf.select(trainer.config, "actor_rollout_ref.rollout.engine_kwargs.vllm.reasoning_parser", default=None),
            tokenizer=trainer.tokenizer,  # type: ignore[arg-type]
        ),
        additional_histories=trainer.extra_config.get("additional_histories", False),
    )


def _build_tool_submit(trainer: "ExperimentTrainer", client: InferenceClient, stage: RolloutStage) -> BaseAgent:
    return ToolSubmitAgent(
        client=client,
        prompt_config=trainer.prompt_config,
        decomp_config=trainer.build_decomp_config(stage),
        tool_parser=build_native_parser(
            tool_parser=trainer.config.actor_rollout_ref.rollout.multi_turn.format,
            reasoning_parser=OmegaConf.select(trainer.config, "actor_rollout_ref.rollout.engine_kwargs.vllm.reasoning_parser", default=None),
            tokenizer=trainer.tokenizer,  # type: ignore[arg-type]
        ),
        additional_histories=trainer.extra_config.get("additional_histories", False),
    )


AGENT_REGISTRY: dict[str, AgentSpec] = {
    AgentKey.DUMMY: AgentSpec(DummyAgent, _build_dummy),
    AgentKey.MARKER: AgentSpec(MarkerAgent, _build_marker),
    AgentKey.JSON: AgentSpec(JsonAgent, _build_json),
    AgentKey.REGEX: AgentSpec(RegexAgent, _build_regex),
    AgentKey.PERST: AgentSpec(PersistentAgent, _build_perst),
    AgentKey.NATIVE_PERST: AgentSpec(NativePersistentAgent, _build_native_perst),
    AgentKey.TOOL_STATELESS: AgentSpec(ToolStatelessAgent, _build_tool_stateless),
    AgentKey.TOOL_PERSISTENT: AgentSpec(ToolPersistentAgent, _build_tool_persistent),
    AgentKey.TOOL_SUBMIT: AgentSpec(ToolSubmitAgent, _build_tool_submit),
}


def build_agent(trainer: "ExperimentTrainer", client: InferenceClient, stage: RolloutStage, key: str) -> BaseAgent:
    if key not in AGENT_REGISTRY:
        raise ValueError(f"Unknown agent {key!r}; expected one of {sorted(AGENT_REGISTRY)}.")
    return AGENT_REGISTRY[key].build(trainer, client, stage)


# Will it cause the GC to not remove stale agents from the same process, since they all will share the same 
# instance of the parser? or no such issue?
@functools.lru_cache(maxsize=None)
def build_native_parser(tool_parser: str, reasoning_parser: str | None, tokenizer):
    """Cached: constructing the vLLM tool/reasoning parsers is the one expensive part of
    building a task, and verl builds a task per rollout."""
    return build_tool_parser(tool_parser=tool_parser, tokenizer=tokenizer, reasoning_parser=reasoning_parser)
