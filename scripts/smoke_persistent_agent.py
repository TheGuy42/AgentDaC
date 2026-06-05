from __future__ import annotations

import argparse
import asyncio
import pathlib
import sys

MODULE_DIR = pathlib.Path(__file__).parent.parent.resolve()
if str(MODULE_DIR) not in sys.path:
    sys.path.append(str(MODULE_DIR))

from src.agents import PersistentAgent
from src.configs import PromptConfig

from smoke_agent_runner import (
    SMOKE_PROMPTS,
    build_client,
    build_decomp_config,
    run_smoke_suite,
)


def build_persistent_prompt_config() -> PromptConfig:
    return PromptConfig(
        mode="path",
        system_root="config_files/prompts/perst/v2_root.txt",
        system_inter="config_files/prompts/perst/v2_inter.txt",
        system_leaf="config_files/prompts/perst/v2_leaf.txt",
        tasks_depleted=None,
    )


async def main() -> int:
    parser = argparse.ArgumentParser(description="PersistentAgent smoke test")
    parser.add_argument("--base-url", default="http://0.0.0.0:8000/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--max-completion-tokens", type=int, default=1024)
    args = parser.parse_args()

    kwargs = {
        "max_completion_tokens": args.max_completion_tokens,
    }
    
    if "Qwen3" in args.model:
        # disable "thinking" for Qwen3 models
        kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}


    return await run_smoke_suite(
        agent_name="PersistentAgent",
        agent_factory=lambda: PersistentAgent(
            openai_client=build_client(args.base_url, args.api_key),
            model_name=args.model,
            prompt_config=build_persistent_prompt_config(),
            decomp_config=build_decomp_config(max_depth=1, max_tasks=3, max_rounds=5),
        ),
        prompts=SMOKE_PROMPTS,
        verbose=True,
        **kwargs,
    )


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
