"""Live smoke test: confirm where a vLLM ``ChatCompletion`` carries token ids.

The whole AgentLightning training path depends on ``prompt_token_ids`` /
``response_token_ids`` being present on each completion (VERL drops triplets without
them). The exact location varies across vLLM / openai-sdk versions, so run this once
against your VERL-managed (or any) vLLM OpenAI endpoint and confirm
``src.utils.convert._extract_token_ids`` finds them.

Usage:
    python scripts/smoke_token_ids.py --base-url http://127.0.0.1:8000/v1 --model <served-model>

If it prints non-empty prompt/response token-id lengths, the conversion path is wired
correctly. If not, inspect the dumped object keys and extend ``_extract_token_ids``.
"""

from __future__ import annotations

import argparse
import asyncio
import pathlib
import sys

MODULE_DIR = pathlib.Path(__file__).parent.parent.resolve()
if str(MODULE_DIR) not in sys.path:
    sys.path.append(str(MODULE_DIR))

from openai import AsyncOpenAI  # noqa: E402


async def main() -> None:
    parser = argparse.ArgumentParser(description="vLLM token-id smoke test")
    parser.add_argument("--base-url", default="http://0.0.0.0:8000/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--prompt", default="What is 2 + 2? Answer in one word.")
    args = parser.parse_args()

    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key)
    resp = await client.chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": args.prompt}],
        max_completion_tokens=16,
        extra_body={"return_token_ids": True},
    )

    print("=== top-level response extra fields ===")
    print(sorted((resp.model_extra or {}).keys()))
    print("=== choice extra fields ===")
    print(sorted((resp.choices[0].model_extra or {}).keys()))
    
    print(resp.prompt_token_ids)
    print(resp.choices[0].token_ids)

    prompt_ids = resp.prompt_token_ids
    response_ids = resp.choices[0].token_ids

    print(f"\n_extract_token_ids -> prompt={len(prompt_ids)} tokens, response={len(response_ids)} tokens")
    if prompt_ids and response_ids:
        print("OK: token ids found — conversion path is wired correctly.")
    else:
        print("FAIL: token ids missing — inspect the dumped keys above and extend _extract_token_ids.")
        print("\nfull response dump:\n", resp.model_dump_json(indent=2)[:4000])


if __name__ == "__main__":
    asyncio.run(main())
