from __future__ import annotations

import argparse
import asyncio
import pathlib
import sys

MODULE_DIR = pathlib.Path(__file__).parent.parent.resolve()
if str(MODULE_DIR) not in sys.path:
    sys.path.append(str(MODULE_DIR))

from src.agents import MarkerAgent

from smoke_agent_runner import (
    COMMON_SMOKE_PROMPTS,
    build_client,
    build_decomp_config,
    build_marker_prompt_config,
    run_smoke_suite,
)


async def main() -> int:
    parser = argparse.ArgumentParser(description="MarkerAgent smoke test")
    parser.add_argument("--base-url", default="http://0.0.0.0:8000/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--max-completion-tokens", type=int, default=256)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    return await run_smoke_suite(
        agent_name="MarkerAgent",
        agent_factory=lambda: MarkerAgent(
            openai_client=build_client(args.base_url, args.api_key),
            model_name=args.model,
            prompt_config=build_marker_prompt_config(),
            decomp_config=build_decomp_config(max_depth=1, max_tasks=1, max_rounds=2),
        ),
        prompts=COMMON_SMOKE_PROMPTS,
        max_completion_tokens=args.max_completion_tokens,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))