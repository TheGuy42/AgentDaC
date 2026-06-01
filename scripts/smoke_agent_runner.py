from __future__ import annotations

import pathlib
import random
from types import SimpleNamespace
from typing import Callable, Sequence, cast
import sys

import agentlightning as agl
from openai import AsyncOpenAI

from src.agents.base import BaseAgent
from src.aliases import UserMessage, Response
from src.configs import DecompConfig, PromptConfig
from src.utils.convert import convert_trajectory

REPO_ROOT = pathlib.Path(__file__).parent.parent.resolve()

SmokePrompt = tuple[str, str]

COMMON_SMOKE_PROMPTS: list[SmokePrompt] = [
    ("arithmetic", "Compute 37 * 48 and give the exact result."),
    ("factorization", "Find the prime factorization of 462 and verify the multiplication."),
    ("debugging", "Give two concrete debugging steps for a failing API smoke test."),
]

PERSISTENT_SMOKE_PROMPTS: list[SmokePrompt] = [
    ("fresh-helper arithmetic", "Use a fresh helper if it helps. Compute 37 * 48 and give the exact result."),
    ("fresh-helper factorization", "If you want to delegate, first create a fresh helper, then find the prime factorization of 462."),
    ("debugging", "Give two concrete debugging steps for a failing API smoke test."),
]


def repo_path(*parts: str) -> str:
    return str(REPO_ROOT.joinpath(*parts))


def build_dummy_rollout(agent_name: str, prompt_index: int) -> SimpleNamespace:
    return SimpleNamespace(
        rollout_id=f"smoke-{agent_name.lower()}-{prompt_index}",
        attempt=SimpleNamespace(attempt_id=str(prompt_index)),
    )


def build_client(base_url: str, api_key: str) -> AsyncOpenAI:
    return AsyncOpenAI(base_url=base_url, api_key=api_key)


def build_decomp_config(*, max_depth: int = 1, max_tasks: int = 1, max_rounds: int = 2) -> DecompConfig:
    return DecompConfig(max_depth=max_depth, max_tasks=max_tasks, max_rounds=max_rounds)


def build_marker_prompt_config() -> PromptConfig:
    return PromptConfig(
        mode="path",
        system_root=repo_path("config_files", "prompts", "gilad", "v2_root.txt"),
        system_inter=repo_path("config_files", "prompts", "gilad", "v2_inter.txt"),
        system_leaf=repo_path("config_files", "prompts", "gilad", "v2_leaf.txt"),
        tasks_depleted=repo_path("config_files", "prompts", "depleted", "v2_depleted.txt"),
    )


def build_regex_prompt_config() -> PromptConfig:
    return PromptConfig(
        mode="path",
        system_root=repo_path("config_files", "prompts", "regex", "v2_root.txt"),
        system_inter=None,
        system_leaf=repo_path("config_files", "prompts", "regex", "v2_leaf.txt"),
        tasks_depleted=None,
    )


def build_json_prompt_config() -> PromptConfig:
    return PromptConfig(
        mode="path",
        system_root=repo_path("config_files", "prompts", "json", "v2_root.txt"),
        system_inter=None,
        system_leaf=repo_path("config_files", "prompts", "json", "v2_leaf.txt"),
        tasks_depleted=None,
    )


PERSISTENT_ROOT_PROMPT = """You are a highly capable and truthful AI assistant that excels at logical reasoning.

You may break complex tasks into smaller sub-tasks. For this agent, the valid actions are:
Action: think | issue_fresh_task | issue_task | answer
Text: <content>

Use issue_fresh_task when you want to create a brand-new helper for the first delegated step.
Use issue_task when you want to send another task to the current helper.
Use answer when you are ready to respond directly.
Return exactly one action/text block and keep the format exact.
"""

PERSISTENT_LEAF_PROMPT = """You are a leaf helper. Do not delegate further.

Action: answer
Text: <content>
"""

PERSISTENT_DEPLETED_PROMPT = """All subtasks are exhausted. Respond directly.

Action: answer
Text: <content>
"""


def build_persistent_prompt_config() -> PromptConfig:
    return PromptConfig(
        mode="text",
        system_root=PERSISTENT_ROOT_PROMPT,
        system_inter=None,
        system_leaf=PERSISTENT_LEAF_PROMPT,
        tasks_depleted=PERSISTENT_DEPLETED_PROMPT,
    )


async def run_smoke_suite(
    *,
    agent_name: str,
    agent_factory: Callable[[], BaseAgent],
    prompts: Sequence[SmokePrompt],
    max_completion_tokens: int,
    verbose: bool,
) -> int:
    failures = 0

    for index, (label, prompt) in enumerate(prompts, start=1):
        agent = agent_factory()
        print(f"\n=== {agent_name} prompt {index}: {label} ===")
        print(prompt)

        try:
            trajectory = await agent.chat(
                UserMessage(role="user", content=prompt),
                verbose=verbose,
                max_completion_tokens=max_completion_tokens,
                
            )

            final_message = trajectory.messages()[-1]
            final_response = trajectory.messages_and_responses[-1]

            if not isinstance(final_response, Response):
                raise ValueError("Final response is not a Response object")

            raw_content = final_message.get("content")
            parsed_answer = agent.parse_answer(final_message)

            trajectory.reward = random.uniform(-1.0, 1.0)
            rollout = cast(agl.AttemptedRollout, build_dummy_rollout(agent_name, index))
            spans = convert_trajectory(trajectory, rollout)

            finish_reason = final_response.choices[0].finish_reason

            print("assistant raw content:")
            print(raw_content)
            print("assistant parsed answer:")
            print(parsed_answer)
            print(f"finish_reason: {finish_reason}")
            print(f"convert_trajectory: OK ({len(spans)} spans, reward={trajectory.reward:.4f})")
            print(f"metrics: {trajectory.metrics}")
        except Exception as exc:
            failures += 1
            print(f"FAIL: {exc}")

    return failures