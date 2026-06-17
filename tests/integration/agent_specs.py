"""Shared scaffolding for the live agent smoke tests.

Consolidates the four near-identical ``scripts/smoke_*.py`` runners into one registry. Each
:class:`AgentSpec` knows how to build its agent (prompt + decomposition config) and the per-call
chat kwargs. The actual assertions live in :func:`run_agent_on_prompt`.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Any, Callable

import agentlightning as agl
from openai import AsyncOpenAI

from src.agents import BaseAgent, JsonAgent, MarkerAgent, PersistentAgent, RegexAgent
from src.aliases import Response, UserMessage
from src.configs import DecompConfig, PromptConfig
from src.custom import VerlAdapter, convert_trajectory

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def repo_path(*parts: str) -> str:
    return str(REPO_ROOT.joinpath(*parts))


# A small, deterministic prompt set. Kept short so live runs stay cheap.
SMOKE_PROMPTS: list[tuple[str, str]] = [
    (
        "prime factorization",
        "Find the complete prime factorization of 1,000,081. Verify your factorization by "
        "multiplying the factors back to confirm they equal the original number.",
    ),
    (
        "cryptarithmetic puzzle",
        "Solve the cryptarithmetic puzzle SEND + MORE = MONEY, where each unique letter represents "
        "a unique digit (0-9). Use systematic logical deduction and verify the complete solution.",
    ),
]


def _build_client(base_url: str, api_key: str) -> AsyncOpenAI:
    return AsyncOpenAI(base_url=base_url, api_key=api_key)


def _chat_kwargs(model: str, max_completion_tokens: int) -> dict[str, Any]:
    # `return_token_ids` is required for `convert_trajectory` to recover prompt/response token ids.
    extra_body: dict[str, Any] = {"return_token_ids": True}
    if "Qwen3" in model:  # disable "thinking" traces for Qwen3 models
        extra_body["chat_template_kwargs"] = {"enable_thinking": False}
    return {"max_completion_tokens": max_completion_tokens, "extra_body": extra_body}


@dataclass(frozen=True)
class AgentSpec:
    name: str
    build_agent: Callable[[AsyncOpenAI, str], BaseAgent]


def _regex_agent(client: AsyncOpenAI, model: str) -> BaseAgent:
    return RegexAgent(
        openai_client=client,
        model_name=model,
        prompt_config=PromptConfig(
            mode="path",
            system_root=repo_path("config_files", "prompts", "regex", "v2_root.txt"),
            system_leaf=repo_path("config_files", "prompts", "regex", "v2_leaf.txt"),
        ),
        decomp_config=DecompConfig(max_depth=1, max_tasks=1, max_rounds=2),
    )


def _json_agent(client: AsyncOpenAI, model: str) -> BaseAgent:
    return JsonAgent(
        openai_client=client,
        model_name=model,
        prompt_config=PromptConfig(
            mode="path",
            system_root=repo_path("config_files", "prompts", "json", "v2_root.txt"),
            system_leaf=repo_path("config_files", "prompts", "json", "v2_leaf.txt"),
        ),
        decomp_config=DecompConfig(max_depth=1, max_tasks=1, max_rounds=2),
    )


def _marker_agent(client: AsyncOpenAI, model: str) -> BaseAgent:
    return MarkerAgent(
        openai_client=client,
        model_name=model,
        prompt_config=PromptConfig(
            mode="path",
            system_root=repo_path("config_files", "prompts", "gilad", "v2_root.txt"),
            system_inter=repo_path("config_files", "prompts", "gilad", "v2_inter.txt"),
            system_leaf=repo_path("config_files", "prompts", "gilad", "v2_leaf.txt"),
            tasks_depleted=repo_path("config_files", "prompts", "depleted", "v2_depleted.txt"),
        ),
        decomp_config=DecompConfig(max_depth=1, max_tasks=1, max_rounds=2),
    )


def _persistent_agent(client: AsyncOpenAI, model: str) -> BaseAgent:
    return PersistentAgent(
        openai_client=client,
        model_name=model,
        prompt_config=PromptConfig(
            mode="path",
            system_root=repo_path("config_files", "prompts", "perst", "v2_root.txt"),
            system_inter=repo_path("config_files", "prompts", "perst", "v2_inter.txt"),
            system_leaf=repo_path("config_files", "prompts", "perst", "v2_leaf.txt"),
        ),
        decomp_config=DecompConfig(max_depth=1, max_tasks=3, max_rounds=5),
    )


AGENT_SPECS: list[AgentSpec] = [
    AgentSpec("regex", _regex_agent),
    AgentSpec("json", _json_agent),
    AgentSpec("marker", _marker_agent),
    AgentSpec("persistent", _persistent_agent),
]


def check_metric_invariants(metrics: dict[str, Any]) -> None:
    """Validate the ``{time}_{scope}_{quantity}`` metric axes after a ``chat()``.

    Invariants that must hold regardless of model output:
      - subtree counts include the agent's own direct work, so ``subtree >= direct``;
      - ``total`` accumulates over all ``chat()`` calls, so ``total >= latest``.
    """
    for key in metrics:
        if key.startswith("total_direct_"):
            subtree_key = f"total_subtree_{key[len('total_direct_'):]}"
            if subtree_key in metrics:
                assert metrics[subtree_key] >= metrics[key], f"{subtree_key} < {key}"

    for key in metrics:
        if key.startswith("latest_"):
            total_key = "total_" + key[len("latest_"):]
            if total_key in metrics and isinstance(metrics[key], (int, float)):
                assert metrics[total_key] >= metrics[key], f"{total_key} < {key}"


async def run_agent_on_prompt(
    *,
    agent: BaseAgent,
    prompt: str,
    rollout: agl.AttemptedRollout,
    reward: float,
    model: str,
    max_completion_tokens: int,
) -> None:
    """Drive one agent on one prompt and assert the trajectory converts to trainable spans."""
    trajectory = await agent.chat(
        UserMessage(role="user", content=prompt),
        verbose=False,
        **_chat_kwargs(model, max_completion_tokens),
    )

    final = trajectory.messages_and_responses[-1]
    assert isinstance(final, Response), "final item in the trajectory must be a model Response"

    trajectory.reward = reward
    check_metric_invariants(trajectory.metrics)

    spans = convert_trajectory(trajectory, rollout)
    assert spans, "convert_trajectory produced no spans"
    assert spans[-1].name == "agentlightning.annotation", "last span must be the reward span"

    triplets = VerlAdapter().adapt(spans)
    assert triplets, "adapter produced no triplets"
    for t in triplets:
        assert t.prompt.get("token_ids"), "triplet has empty prompt token ids"
        assert t.response.get("token_ids"), "triplet has empty response token ids"


def build_client(base_url: str, api_key: str) -> AsyncOpenAI:
    return _build_client(base_url, api_key)
