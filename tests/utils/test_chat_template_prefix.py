"""Token-level prefix-preservation check for the chat template used by the tool agents.

Why this exists
---------------
`convert_trajectory` (src/custom/convert.py) rebuilds the RL training sequence from the
per-turn prompt tokens and REQUIRES that each turn's prompt is a *token*-prefix of the next
(append-only). We re-tokenize the whole message list every turn via
`tokenizer.apply_chat_template(...)` (see `VerlClient._tokenize`), so the template must be
strictly token-prefix-preserving across turns.

For Qwen3.5 with ``enable_thinking=True`` it is NOT, and TRL's auto-patch
(`get_training_chat_template`, used by `src/utils/chat_template.resolve_chat_template`) does
not fix it (its `is_chat_template_prefix_preserving` check is, per its own comment, "not
always accurate"). The failure mode:

  * the *generation* prompt for an assistant turn ends with ``<|im_start|>assistant\\n<think>\\n``
    (the model then reasons inside that opened block);
  * when that same turn becomes *history*, the template instead renders an EMPTY
    ``<think>\\n\\n</think>\\n\\n`` block and appends the generated text as plain content
    -- discarding the real reasoning structure and flipping the boundary token
    (``\\n`` -> ``\\n\\n``).

So turn *t*'s prompt is not a prefix of turn *t+1*'s prompt, `convert_trajectory` refuses,
and every multi-turn (delegating) rollout is dropped as degenerate.

Using this file
---------------
* Run ``pytest -s src/utils/tests/test_chat_template_prefix.py`` to SEE the exact decoded
  divergence (the main test is expected to FAIL against the current template).
* To validate your own fixed template, point the env var at it and re-run -- the test passes
  once it is token-prefix-preserving:
      ``CANDIDATE_CHAT_TEMPLATE=/path/to/template.jinja pytest -s src/utils/tests/test_chat_template_prefix.py``
* `check_prefix_preserving(...)` is the reusable checker; `VERBATIM_TEMPLATE` shows the shape
  of a prefix-preserving template (assistant turns emitted verbatim, no think-block surgery).
"""

from __future__ import annotations

import os

import pytest

import argparse
import pathlib
import sys
from typing import Any

# set pythonpath to the main module directory
module_dir = pathlib.Path(__file__).parent.parent.parent.resolve()
if str(module_dir) not in sys.path:
    sys.path.append(str(module_dir))

MODEL = "Qwen/Qwen3.5-4B"
TEMPLATE_KWARGS = {"enable_thinking": True}

# The DAC tool schema (single string arg), advertised prompt-side like the real run.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "create_new_sub_agent",
            "description": "Delegate a self-contained sub-task to a fresh sub-agent.",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string", "description": "the sub-task"}},
                "required": ["text"],
                "additionalProperties": False,
            },
        },
    }
]


def _delegate_content() -> str:
    # Raw generation: begins AFTER the prompt-injected "<think>\n", so it starts with
    # reasoning text and carries only the CLOSING </think>, then the tool call.
    return (
        "Let me break this position down before committing to a move.\n"
        "</think>\n\n"
        "<tool_call>\n<function=create_new_sub_agent>\n<parameter=text>\n"
        "Find the best move for White in r1bqkb1r/1p1p1ppp/p1n2n2/2pQ4/2B1P3/5N2/PPP2PPP/RNB1K2R w KQkq - 1 7. "
        "Reply with a single UCI move.\n"
        "</parameter>\n</function>\n</tool_call>"
    )


def _answer_content() -> str:
    return "The sub-agent's move checks out against the position.\n</think>\n\nThe best move is \\boxed{e2e4}."


def _conversation() -> list[dict]:
    """A minimal DAC tool-agent trajectory: reason->delegate, then reason->answer."""
    return [
        {"role": "system", "content": "You solve chess tasks; keep your reasoning brief."},
        {"role": "user", "content": "Task: find the best legal move for the side to move in the given position."},
        {"role": "user", "content": "[status] rounds remaining: 5. Delegate a sub-task or write your final answer."},
        {"role": "assistant", "content": _delegate_content()},
        {"role": "tool", "content": "e2e4"},
        {"role": "user", "content": "[status] rounds remaining: 4. Delegate a sub-task or write your final answer."},
        {"role": "assistant", "content": _answer_content()},
    ]


# A trivially prefix-preserving template: every message emitted verbatim, generation prompt
# is just the assistant header. Shows the *shape* a fixed template needs (assistant content,
# including its reasoning, is reproduced verbatim in history -- no empty-block injection).
VERBATIM_TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


def check_prefix_preserving(tokenizer, messages, tools, **template_kwargs):
    """Return ``None`` if the template is token-prefix-preserving across generation turns,
    else a diagnostic dict describing the first divergence.

    Mirrors the invariant `convert_trajectory` enforces: for each generation turn, the
    re-tokenized prompt must extend the previous turn's prompt (append-only).
    """
    from verl.utils.tokenizer import normalize_token_ids

    def prompt_ids(upto: int) -> list[int]:
        return normalize_token_ids(
            tokenizer.apply_chat_template(
                messages[:upto],
                tools=tools,
                add_generation_prompt=True,
                tokenize=True,
                **template_kwargs,
            )
        )

    assistant_turns = [i for i, m in enumerate(messages) if m["role"] == "assistant"]
    prev_ids: list[int] | None = None
    prev_turn = None
    for turn, idx in enumerate(assistant_turns):
        cur_ids = prompt_ids(idx)
        if prev_ids is not None and cur_ids[: len(prev_ids)] != prev_ids:
            j = next(
                (k for k in range(min(len(cur_ids), len(prev_ids))) if cur_ids[k] != prev_ids[k]),
                min(len(cur_ids), len(prev_ids)),
            )
            return {
                "turn": turn,
                "prev_turn": prev_turn,
                "diverge_token_index": j,
                "prev_render": tokenizer.decode(prev_ids[max(0, j - 6) : j + 10]),
                "cur_render": tokenizer.decode(cur_ids[max(0, j - 6) : j + 10]),
            }
        prev_ids, prev_turn = cur_ids, turn
    return None


def _load_tokenizer():
    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL)
    except Exception as exc:  # network/cache unavailable
        pytest.skip(f"{MODEL} tokenizer unavailable: {exc}")

    candidate = os.environ.get("CANDIDATE_CHAT_TEMPLATE")
    if candidate:
        tokenizer.chat_template = open(candidate).read()
        return tokenizer, f"candidate template {candidate}"

    from src.utils.chat_template import resolve_chat_template

    tokenizer.chat_template = resolve_chat_template(MODEL, None)  # the current TRL auto-patch
    return tokenizer, "TRL auto-patched default template"


def test_checker_passes_for_verbatim_template():
    """Sanity: the checker returns None for a genuinely prefix-preserving (verbatim) template,
    and this doubles as the target shape for a fixed template."""
    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL)
    except Exception as exc:
        pytest.skip(f"{MODEL} tokenizer unavailable: {exc}")
    tokenizer.chat_template = VERBATIM_TEMPLATE
    assert check_prefix_preserving(tokenizer, _conversation(), TOOLS, **TEMPLATE_KWARGS) is None


def test_template_is_token_prefix_preserving():
    """The chat template must be token-prefix-preserving across turns, or convert_trajectory
    drops every multi-turn rollout as degenerate.

    EXPECTED TO FAIL against Qwen3.5 + enable_thinking with the current auto-patch; it will
    pass once a prefix-preserving template is supplied (directly or via CANDIDATE_CHAT_TEMPLATE).
    """
    tokenizer, source = _load_tokenizer()
    diag = check_prefix_preserving(tokenizer, _conversation(), TOOLS, **TEMPLATE_KWARGS)
    if diag is not None:
        pytest.fail(
            f"Chat template ({source}) is NOT token-prefix-preserving; convert_trajectory will "
            f"reject multi-turn rollouts as degenerate.\n"
            f"  first divergence at turn {diag['turn']} (vs turn {diag['prev_turn']}), "
            f"token #{diag['diverge_token_index']}\n"
            f"  generation opener   (turn {diag['prev_turn']}): {diag['prev_render']!r}\n"
            f"  history re-render    (turn {diag['turn']}): {diag['cur_render']!r}"
        )