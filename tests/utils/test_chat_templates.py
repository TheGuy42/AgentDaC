"""Branch-coverage tests for the chat templates in ``config_files/templates``.

Why this exists
---------------
Each model gets two hand-maintained variants:

* ``<name>.original.jinja``   -- the model's own template, with ONLY the ART fix applied.
* ``<name>.preserving.jinja`` -- always token-prefix-preserving, and also ART-safe.

Two contracts follow, and both are easy to break silently:

* ``.original`` must render **byte-identically to the model's own template** everywhere except the
  final (trainable) assistant message. A regression here is invisible until a rollout diverges.
* ``.preserving`` must be **token**-prefix-preserving for every input, because verl re-tokenizes the
  whole conversation each turn and drops multi-turn rollouts whose prompt is not append-only.

`tests/utils/test_chat_template_prefix.py` checks the prefix property for one canonical shape. It
passes both before and after every version of these fixes, so it is necessary but not sufficient.
This module instead sweeps the templates' own branch structure -- tools/no-tools, every assistant
content shape, tool-call arities and argument types, consecutive tool messages, multimodal content
parts, and every `enable_thinking` / `preserve_thinking` value -- and compares against the default.

Everything is asserted against the model's own template, so "before vs after" is checked on every
case rather than assumed.
"""

from __future__ import annotations

import json
import pathlib
import sys
from typing import Any

import pytest

MODULE_DIR = pathlib.Path(__file__).resolve().parents[2]
if str(MODULE_DIR) not in sys.path:
    sys.path.append(str(MODULE_DIR))

TEMPLATE_DIR = MODULE_DIR / "config_files" / "templates"

# name -> (hf id, family)
# family "own": the model generates its own <think>; the generation prompt injects nothing.
# family "pre": the generation prompt pre-opens '<think>\n', so content has no opening tag.
MODELS: dict[str, tuple[str, str]] = {
    "Qwen3": ("Qwen/Qwen3-14B", "own"),
    "Qwen3-Thinking-2507": ("Qwen/Qwen3-4B-Thinking-2507", "pre"),
    "Qwen3.5": ("Qwen/Qwen3.5-4B", "pre"),
    "Qwen3.6": ("Qwen/Qwen3.6-27B", "pre"),
}
VARIANTS = ("original", "preserving")

# Only Qwen3.6 understands preserve_thinking; only these two route content through render_content.
SUPPORTS_PRESERVE_THINKING = {"Qwen3.6"}
SUPPORTS_MULTIMODAL = {"Qwen3.5", "Qwen3.6"}
# Qwen3-Thinking-2507 always pre-opens the think block; it has no thinking-off mode.
SUPPORTS_THINKING_OFF = {"Qwen3", "Qwen3.5", "Qwen3.6"}

TOOLS: list[dict[str, Any]] = [
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

# Assistant content shapes, per family. These are the generation shapes actually observed plus the
# degenerate ones that broke the templates historically.
SHAPES: dict[str, dict[str, str]] = {
    "own": {
        "canonical": "<think>\n{r}\n</think>\n\n{a}",
        "empty-think": "<think>\n</think>\n\n{a}",
        "extra-newline": "<think>\n\n{r}\n</think>\n\n{a}",
        "no-blank-after-close": "<think>\n{r}\n</think>{a}",
        "no-newlines": "<think>{r}</think>{a}",
        "empty-answer": "<think>\n{r}\n</think>\n\n",
        "doubled-think": "<think>\n{r}\n</think>\n\n<think>\n{r}\n</think>\n\n{a}",
        "unterminated": "<think>\n\n{r} ran out of tokens",
        "trailing-ws": "<think>\n{r}\n</think>\n\n{a}\n\n",
    },
    "pre": {
        "canonical": "{r}\n</think>\n\n{a}",
        "empty-think": "\n</think>\n\n{a}",
        "extra-newline": "\n{r}\n</think>\n\n{a}",
        "no-blank-after-close": "{r}\n</think>{a}",
        "no-newlines": "{r}</think>{a}",
        "empty-answer": "{r}\n</think>\n\n",
        "unterminated": "\n{r} ran out of tokens",
        "trailing-ws": "{r}\n</think>\n\n{a}\n\n",
        "leading-blank": "\n\n{r}\n</think>\n\n{a}",
    },
}
# Content a model produces when thinking is disabled: no think tags at all.
NO_THINK_SHAPES = {"answer-only": "{a}", "leading-newline": "\n{a}", "empty": ""}


def _shape(family: str, key: str, i: int) -> str:
    table = SHAPES[family] if key in SHAPES[family] else NO_THINK_SHAPES
    return table[key].format(r=f"reasoning{i}", a=f"answer{i}")


def normalize_token_ids(out: Any) -> list[int]:
    """Flatten apply_chat_template(tokenize=True) across transformers 4 and 5.

    v4 returns list[int]; v5 may return a BatchEncoding/mapping carrying ``input_ids``. Kept local so
    this module does not depend on verl, which is only installed in the verl environment.
    """
    ids = out
    if isinstance(ids, dict):
        ids = ids.get("input_ids", ids)
    elif hasattr(ids, "input_ids"):
        ids = ids.input_ids
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if isinstance(ids, (list, tuple)) and len(ids) == 1 and isinstance(ids[0], (list, tuple)):
        ids = list(ids[0])
    return [int(t) for t in ids]


@pytest.fixture(scope="module")
def tokenizers() -> dict[str, Any]:
    from transformers import AutoTokenizer

    loaded = {}
    for name, (hf_id, _) in MODELS.items():
        try:
            loaded[name] = AutoTokenizer.from_pretrained(hf_id, local_files_only=True)
        except Exception as exc:  # tokenizer not cached in this environment
            loaded[name] = exc
    return loaded


def _tokenizer(tokenizers: dict[str, Any], name: str):
    tok = tokenizers[name]
    if isinstance(tok, Exception):
        pytest.skip(f"{MODELS[name][0]} tokenizer unavailable: {tok}")
    return tok


def default_template(tok) -> str:
    template = tok.chat_template
    if isinstance(template, list):  # legacy multi-template format
        template = template[0]["template"]
    assert isinstance(template, str) and template, "model has no chat template"
    return template


def variant_template(name: str, variant: str) -> str:
    path = TEMPLATE_DIR / f"{name}.{variant}.jinja"
    assert path.exists(), f"missing template: {path}"
    return path.read_text(encoding="utf-8")


def render(tok, template: str, messages: list[dict], **kwargs) -> str:
    from transformers.utils.chat_template_utils import render_jinja_template

    return render_jinja_template(conversations=[messages], chat_template=template, **kwargs)[0][0]


def encode(tok, template: str, messages: list[dict], **kwargs) -> list[int]:
    tok.chat_template = template
    return normalize_token_ids(tok.apply_chat_template(messages, **kwargs))


def settings_for(name: str) -> list[dict[str, Any]]:
    """Every flag combination the template can observe, including 'not passed at all' (the ART path)."""
    out = []
    thinking = [True, False, None] if name in SUPPORTS_THINKING_OFF else [True, None]
    preserve = [True, False, None] if name in SUPPORTS_PRESERVE_THINKING else [None]
    for et in thinking:
        for pt in preserve:
            kw: dict[str, Any] = {}
            if et is not None:
                kw["enable_thinking"] = et
            if pt is not None:
                kw["preserve_thinking"] = pt
            out.append(kw)
    return out


# --------------------------------------------------------------------------------------------
# Conversations, chosen to hit the templates' branches: the last_query_index gate (assistant before
# and after the final user turn), tool-call arity and argument types, consecutive tool messages,
# a <tool_response>-wrapped user turn, and a trailing non-assistant message.
# --------------------------------------------------------------------------------------------
def conversations(family: str, shape: str) -> list[tuple[str, list[dict]]]:
    a1, a2 = _shape(family, shape, 1), _shape(family, shape, 2)
    sys_msg = {"role": "system", "content": "You solve tasks."}
    tc_one = [{"type": "function", "function": {"name": "create_new_sub_agent", "arguments": {"text": "sub"}}}]
    tc_two = tc_one + [{"type": "function", "function": {"name": "create_new_sub_agent", "arguments": {"text": "other"}}}]
    tc_str = [{"type": "function", "function": {"name": "create_new_sub_agent", "arguments": '{"text": "raw"}'}}]
    return [
        ("single-turn", [sys_msg, {"role": "user", "content": "q1"}, {"role": "assistant", "content": a1}]),
        ("no-system", [{"role": "user", "content": "q1"}, {"role": "assistant", "content": a1}]),
        (
            "two-turn",
            [sys_msg, {"role": "user", "content": "q1"}, {"role": "assistant", "content": a1},
             {"role": "user", "content": "q2"}, {"role": "assistant", "content": a2}],
        ),
        (
            "three-turn",
            [sys_msg, {"role": "user", "content": "q1"}, {"role": "assistant", "content": a1},
             {"role": "user", "content": "q2"}, {"role": "assistant", "content": a2},
             {"role": "user", "content": "q3"}, {"role": "assistant", "content": a1}],
        ),
        (
            "tool-call-single",
            [sys_msg, {"role": "user", "content": "q1"}, {"role": "assistant", "content": a1, "tool_calls": tc_one},
             {"role": "tool", "content": "result"}, {"role": "assistant", "content": a2}],
        ),
        (
            "tool-call-multiple",
            [sys_msg, {"role": "user", "content": "q1"}, {"role": "assistant", "content": a1, "tool_calls": tc_two},
             {"role": "tool", "content": "r1"}, {"role": "tool", "content": "r2"},
             {"role": "assistant", "content": a2}],
        ),
        (
            "tool-call-string-args",
            [sys_msg, {"role": "user", "content": "q1"}, {"role": "assistant", "content": a1, "tool_calls": tc_str},
             {"role": "tool", "content": "result"}, {"role": "assistant", "content": a2}],
        ),
        (
            "tool-call-last-turn",
            [sys_msg, {"role": "user", "content": "q1"}, {"role": "assistant", "content": a1, "tool_calls": tc_one}],
        ),
        (
            "tool-response-user",
            [sys_msg, {"role": "user", "content": "q1"}, {"role": "assistant", "content": a1},
             {"role": "user", "content": "<tool_response>\nout\n</tool_response>"},
             {"role": "assistant", "content": a2}],
        ),
        (
            "trailing-user",
            [sys_msg, {"role": "user", "content": "q1"}, {"role": "assistant", "content": a1},
             {"role": "user", "content": "q2"}],
        ),
    ]


ALL_SHAPES = sorted(set(SHAPES["own"]) | set(SHAPES["pre"]) | set(NO_THINK_SHAPES))


def _cases(name: str):
    family = MODELS[name][1]
    shapes = [s for s in ALL_SHAPES if s in SHAPES[family] or s in NO_THINK_SHAPES]
    for shape in shapes:
        for conv_id, messages in conversations(family, shape):
            for tools in (None, TOOLS):
                for kwargs in settings_for(name):
                    yield shape, conv_id, tools, kwargs


def _last_assistant_index(messages: list[dict]) -> int | None:
    idx = [i for i, m in enumerate(messages) if m["role"] == "assistant"]
    return idx[-1] if idx else None


def safe_render(tok, template: str, messages: list[dict], **kwargs) -> tuple[str, str]:
    """``("ok", text)`` or ``("raise", ExceptionName)``.

    Some inputs are unsupported by a model's own template -- e.g. Qwen3.5/3.6 call
    ``tool_call.arguments|items`` unconditionally, so string arguments raise there while Qwen3
    handles them. The contract is parity with the default, not absolute success, so both sides are
    compared including their failures.
    """
    try:
        return ("ok", render(tok, template, messages, **kwargs))
    except Exception as exc:
        return ("raise", type(exc).__name__)


# --------------------------------------------------------------------------------------------
# Contract 1: `.original` == the model's own template, everywhere but the final assistant message.
# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(MODELS))
def test_original_matches_default_except_final_assistant(tokenizers, name):
    tok = _tokenizer(tokenizers, name)
    default, ours = default_template(tok), variant_template(name, "original")
    mismatches = []
    for shape, conv_id, tools, kwargs in _cases(name):
        messages = dict(conversations(MODELS[name][1], shape))[conv_id]
        a_kind, a = safe_render(tok, default, messages, tools=tools, **kwargs)
        b_kind, b = safe_render(tok, ours, messages, tools=tools, **kwargs)
        if a_kind != b_kind:
            mismatches.append((shape, conv_id, bool(tools), kwargs, f"default={a_kind} ours={b_kind}"))
            continue
        if a_kind == "raise":
            continue  # unsupported by the model's own template too -- parity is the contract
        if messages[-1]["role"] != "assistant":
            # no trainable final message -> the whole render must match
            if a != b:
                mismatches.append((shape, conv_id, bool(tools), kwargs, "whole render"))
            continue
        cut = lambda s: s[: s.rindex("<|im_start|>assistant")]
        if cut(a) != cut(b):
            mismatches.append((shape, conv_id, bool(tools), kwargs, "prefix"))
    assert not mismatches, f"{len(mismatches)} divergence(s) from the default template, e.g. {mismatches[:5]}"


# --------------------------------------------------------------------------------------------
# Contract 2: `.preserving` is token-prefix-preserving for every input.
# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(MODELS))
def test_preserving_is_token_prefix_preserving(tokenizers, name):
    tok = _tokenizer(tokenizers, name)
    default = default_template(tok)
    template = variant_template(name, "preserving")
    breaks = []
    for shape, conv_id, tools, kwargs in _cases(name):
        messages = dict(conversations(MODELS[name][1], shape))[conv_id]
        if safe_render(tok, default, messages, tools=tools, **kwargs)[0] == "raise":
            continue  # input the model's own template cannot render either
        assistants = [i for i, m in enumerate(messages) if m["role"] == "assistant"]
        prev = None
        for idx in assistants:
            cur = encode(tok, template, messages[:idx], tools=tools, add_generation_prompt=True, **kwargs)
            if prev is not None and cur[: len(prev)] != prev:
                breaks.append((shape, conv_id, bool(tools), kwargs))
                break
            prev = cur
    assert not breaks, f"{len(breaks)} prefix break(s), e.g. {breaks[:5]}"


# --------------------------------------------------------------------------------------------
# Contract 3: under ART's sentinel splice, the trained sequence equals prompt + generated tokens.
# Only the reachable diagonal is asserted: a thinking-on rollout produces content carrying a think
# block, a thinking-off rollout produces content without one. The off-diagonal cannot occur.
# --------------------------------------------------------------------------------------------
def _art_trained_tokens(tok, template: str, messages: list[dict], generated: str, **kwargs) -> list[int]:
    tok.chat_template = template
    original = normalize_token_ids(tok.apply_chat_template(messages, continue_final_message=True, **kwargs))
    sentinel_id = max(set(range(tok.vocab_size)) - set(original))
    stub = list(messages[:-1]) + [{"role": "assistant", "content": tok.decode(sentinel_id)}]
    ids = normalize_token_ids(tok.apply_chat_template(stub, continue_final_message=True, **kwargs))
    gen = tok.encode(generated, add_special_tokens=False)
    start = ids.index(sentinel_id)
    end = start + 1
    # ART rewinds over a template-injected empty think block when the generation opens its own.
    if gen and tok.decode([gen[0]]) == "<think>" and start >= 4 and tok.decode([ids[start - 4]]) == "<think>":
        start -= 4
    ids[start:end] = gen
    return ids


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("name", sorted(MODELS))
def test_art_splice_reproduces_prompt_plus_generated(tokenizers, name, variant):
    tok = _tokenizer(tokenizers, name)
    family = MODELS[name][1]
    template = variant_template(name, variant)
    mismatches = []
    thinking_on = [{}] + ([{"enable_thinking": True}] if True else [])
    for shape in SHAPES[family]:
        generated = _shape(family, shape, 1)
        messages = [{"role": "user", "content": "q1"}, {"role": "assistant", "content": generated}]
        for kwargs in thinking_on:
            prompt = encode(tok, template, messages[:-1], add_generation_prompt=True, **kwargs)
            truth = prompt + tok.encode(generated, add_special_tokens=False)
            got = _art_trained_tokens(tok, template, messages, generated, **kwargs)
            if got != truth:
                mismatches.append((shape, kwargs))
    if name in SUPPORTS_THINKING_OFF:
        for shape in NO_THINK_SHAPES:
            generated = _shape(family, shape, 1)
            messages = [{"role": "user", "content": "q1"}, {"role": "assistant", "content": generated}]
            kwargs = {"enable_thinking": False}
            prompt = encode(tok, template, messages[:-1], add_generation_prompt=True, **kwargs)
            truth = prompt + tok.encode(generated, add_special_tokens=False)
            if _art_trained_tokens(tok, template, messages, generated, **kwargs) != truth:
                mismatches.append((shape, kwargs))
    assert not mismatches, f"{len(mismatches)} token mismatch(es), e.g. {mismatches[:5]}"


# --------------------------------------------------------------------------------------------
# Contract 4: transformers' continue_final_message check never raises. This is the crash that
# killed a training run at the step-1 tokenize call.
# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("name", sorted(MODELS))
def test_no_continue_final_message_crash(tokenizers, name, variant):
    tok = _tokenizer(tokenizers, name)
    default = default_template(tok)
    template = variant_template(name, variant)
    failures = []
    for shape, conv_id, tools, kwargs in _cases(name):
        messages = dict(conversations(MODELS[name][1], shape))[conv_id]
        if messages[-1]["role"] != "assistant" or messages[-1].get("tool_calls"):
            continue
        if safe_render(tok, default, messages, tools=tools, **kwargs)[0] == "raise":
            continue  # input the model's own template cannot render either
        kind, detail = safe_render(tok, template, messages, tools=tools,
                                   continue_final_message=True, **kwargs)
        if kind == "raise":
            failures.append((shape, conv_id, bool(tools), kwargs, detail))
    assert not failures, f"{len(failures)} crash(es), e.g. {failures[:3]}"


# --------------------------------------------------------------------------------------------
# Contract 5: neither variant may alter the generation prompt -- only history serialization.
# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(MODELS))
def test_generation_prompt_identical_to_default(tokenizers, name):
    tok = _tokenizer(tokenizers, name)
    default = default_template(tok)
    messages = [{"role": "user", "content": "q1"}]
    for tools in (None, TOOLS):
        for kwargs in settings_for(name):
            expected = encode(tok, default, messages, tools=tools, add_generation_prompt=True, **kwargs)
            for variant in VARIANTS:
                got = encode(tok, variant_template(name, variant), messages, tools=tools,
                             add_generation_prompt=True, **kwargs)
                assert got == expected, f"{name}.{variant} changed the generation prompt ({kwargs}, tools={bool(tools)})"


# --------------------------------------------------------------------------------------------
# Contract 6: multimodal content parts and the templates' raise_exception paths behave as default.
# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("name", sorted(n for n in MODELS if n in SUPPORTS_MULTIMODAL))
def test_multimodal_and_exception_parity(tokenizers, name, variant):
    tok = _tokenizer(tokenizers, name)
    default, ours = default_template(tok), variant_template(name, variant)
    family = MODELS[name][1]
    a1 = _shape(family, "canonical", 1)
    cases = {
        "image-part": [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "what?"}]},
                       {"role": "assistant", "content": a1}],
        "video-part": [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": "what?"}]},
                       {"role": "assistant", "content": a1}],
        "none-content": [{"role": "user", "content": None}, {"role": "assistant", "content": a1}],
        "system-not-first": [{"role": "user", "content": "q"}, {"role": "system", "content": "s"},
                             {"role": "assistant", "content": a1}],
        "no-user-query": [{"role": "system", "content": "s"}, {"role": "assistant", "content": a1}],
        "no-messages": [],
    }
    for case, messages in cases.items():
        def run(template):
            try:
                return ("ok", render(tok, template, messages, add_vision_id=True))
            except Exception as exc:
                return ("raise", type(exc).__name__)

        d_kind, d_val = run(default)
        o_kind, o_val = run(ours)
        assert d_kind == o_kind, f"{name}.{variant} [{case}]: default={d_kind} ours={o_kind}"
        if d_kind == "ok":
            cut = lambda s: s[: s.rindex("<|im_start|>assistant")] if "<|im_start|>assistant" in s else s
            assert cut(d_val) == cut(o_val), f"{name}.{variant} [{case}]: prefix differs from default"
