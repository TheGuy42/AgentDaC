from __future__ import annotations

import argparse
import asyncio
import pathlib
import sys

MODULE_DIR = pathlib.Path(__file__).parent.parent.resolve()
if str(MODULE_DIR) not in sys.path:
    sys.path.append(str(MODULE_DIR))

from src.agents import JsonAgent
from src.configs import PromptConfig

from smoke_agent_runner import (
    SMOKE_PROMPTS,
    build_client,
    build_decomp_config,
    run_smoke_suite,
    repo_path,
)


def build_json_prompt_config() -> PromptConfig:
    return PromptConfig(
        mode="path",
        system_root=repo_path("config_files", "prompts", "json", "v2_root.txt"),
        system_inter=None,
        system_leaf=repo_path("config_files", "prompts", "json", "v2_leaf.txt"),
        tasks_depleted=None,
    )

async def main() -> int:
    parser = argparse.ArgumentParser(description="JsonAgent smoke test")
    parser.add_argument("--base-url", default="http://0.0.0.0:8000/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--max-completion-tokens", type=int, default=256)
    parser.add_argument("--multi-step", action="store_true", help="Reuse one agent instance across all prompts (multi-step interaction)")
    args = parser.parse_args()

    return await run_smoke_suite(
        agent_name="JsonAgent",
        agent_factory=lambda: JsonAgent(
            openai_client=build_client(args.base_url, args.api_key),
            model_name=args.model,
            prompt_config=build_json_prompt_config(),
            decomp_config=build_decomp_config(max_depth=1, max_tasks=1, max_rounds=2),
        ),
        prompts=SMOKE_PROMPTS,
        max_completion_tokens=args.max_completion_tokens,
        verbose=True,
        reuse_agent=args.multi_step,
    )


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))