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
from src.custom.convert import convert_trajectory

REPO_ROOT = pathlib.Path(__file__).parent.parent.resolve()

SmokePrompt = tuple[str, str]

SMOKE_PROMPTS: list[SmokePrompt] = [
    (
        "cryptarithmetic puzzle",
        "Solve the cryptarithmetic puzzle SEND + MORE = MONEY, where each unique letter represents a unique digit (0-9). Use systematic logical deduction (feel free to delegate aspects to helpers) and verify the complete solution.",
    ),
    (
        "complex prime factorization",
        "Find the complete prime factorization of 1,000,081. Verify your factorization by multiplying the factors back to confirm they equal the original number.",
    ),
    (
        "algorithmic number theory",
        "Find all unique pairs (a,b) where 1 ≤ a ≤ b ≤ 20 and a³ + b³ is also a perfect cube. Verify each solution by computing the cubes and checking if the sum is a perfect cube.",
    ),
    (
        "chess endgame analysis",
        "Analyze the following chess endgame position: White King on e4, White Rook on a1, Black King on h8, Black Rook on h1. It is White's turn. Find the best move that leads to checkmate within 10 moves, or determine if White can force a win. Explain your reasoning step-by-step, considering Black's best defensive resources.",
    ),
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


async def run_smoke_suite(
    *,
    agent_name: str,
    agent_factory: Callable[[], BaseAgent],
    prompts: Sequence[SmokePrompt],
    verbose: bool,
    **kwargs,
) -> int:
    failures = 0

    for index, (label, prompt) in enumerate(prompts, start=1):
        agent = agent_factory()
        print(f"\n\n=== {agent_name} prompt {index}: {label} ===\n\n")
        print(prompt)
        print("\n\n")

        try:
            trajectory = await agent.chat(
                UserMessage(role="user", content=prompt),
                verbose=verbose,
                **kwargs,
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

            print("\n\n")
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
