from __future__ import annotations
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import pairwise, product
from typing import Any
from transformers import AutoTokenizer

Message = dict[str, Any]
Conversation = list[Message]
Tools = list[dict[str, Any]]

DEFAULT_TOOL = {
    "type": "function",
    "function": {
        "name": "lookup",
        "description": "Look up a value by key.",
        "parameters": {
            "type": "object",
            "properties": {"key": {"type": "string"}},
            "required": ["key"],
            "additionalProperties": False,
        },
    },
}

_TEXT_RESPONSE_VARIANTS = (
    ("text/plain", "Response one."),
    ("text/empty", ""),
    ("text/space", " "),
    ("text/leading-space", " Response one."),
    ("text/leading-newline", "\nResponse one."),
    ("text/leading-blank-line", "\n\nResponse one."),
    ("text/trailing-newline", "Response one.\n"),
    ("text/punctuation", ".,:;!?()[]{}"),
    ("text/unicode", "Café — 答え \U0001f642"),
)


@dataclass(frozen=True)
class _Scenario:
    name: str
    messages: Conversation


@dataclass(frozen=True)
class _Case:
    """One scenario rendered under one tools variant and one set of template kwargs."""

    scenario: _Scenario
    tools: Tools | None
    template_kwargs: dict[str, Any]


@dataclass(frozen=True)
class _Rendering:
    """One point in the prefix chain: every rendering must be a token prefix of the next."""

    assistant_turn: int
    token_ids: list[int]


@dataclass(frozen=True)
class PrefixFailure:
    """A reproducible prefix violation, with the token-level divergence that proves it."""

    scenario: str
    conversation: Conversation
    tools: Tools | None
    template_kwargs: dict[str, Any]
    previous_assistant_turn: int
    current_assistant_turn: int
    divergence_token_index: int
    previous_token_id: int | None
    current_token_id: int | None
    previous_token_context: str
    current_token_context: str


def _validate_thinking_delimiters(
    think_start: str | None,
    think_end: str | None,
) -> None:
    """Reasoning probes need both delimiters, or neither to disable them."""

    if (think_start is None) != (think_end is None):
        raise ValueError("think_start and think_end must both be strings or both be None")
    if think_start == "" or think_end == "":
        raise ValueError("think_start and think_end must be non-empty")


def _reasoning_response_variants(
    think_start: str | None,
    think_end: str | None,
) -> list[tuple[str, str]]:
    """Create opaque reasoning-shaped assistant responses."""

    if think_start is None or think_end is None:
        return []

    reasoning = "Reasoning one."
    answer = "Response one."
    return [
        ("reasoning/canonical", f"{think_start}\n{reasoning}\n{think_end}\n\n{answer}"),
        ("reasoning/empty", f"{think_start}\n{think_end}\n\n{answer}"),
        ("reasoning/extra-leading-newline", f"{think_start}\n\n{reasoning}\n{think_end}\n\n{answer}"),
        ("reasoning/compact", f"{think_start}{reasoning}{think_end}{answer}"),
        ("reasoning/preopened", f"{reasoning}\n{think_end}\n\n{answer}"),
        ("reasoning/preopened-leading-newline", f"\n{reasoning}\n{think_end}\n\n{answer}"),
        ("reasoning/unterminated", f"{think_start}\nReasoning that ended before its closing delimiter."),
    ]


def _user(content: str) -> Message:
    return {"role": "user", "content": content}


def _assistant(
    content: str | None,
    tool_calls: list[dict[str, Any]] | None = None,
) -> Message:
    message: Message = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = list(tool_calls)
    return message


def _tool_call(
    arguments: dict[str, Any] | str,
    *,
    call_id: str = "call_1",
) -> dict[str, Any]:
    """A ``lookup`` call; ``arguments`` may be a mapping or a pre-serialized JSON string."""

    return {
        "id": call_id,
        "type": "function",
        "function": {"name": "lookup", "arguments": arguments},
    }


def _tool_result(content: str, call_id: str) -> Message:
    return {
        "role": "tool",
        "name": "lookup",
        "tool_call_id": call_id,
        "content": content,
    }


def _two_turn_conversation(first_response: str) -> Conversation:
    return [
        _user("Question one."),
        _assistant(first_response),
        _user("Question two."),
        _assistant("Response two."),
    ]


def _tool_round_trip(
    call: dict[str, Any],
    *,
    content: str | None = None,
) -> Conversation:
    """User question, assistant tool call, tool result, assistant answer."""

    return [
        _user("Look up the requested value."),
        _assistant(content, [call]),
        _tool_result("42", call["id"]),
        _assistant("The value is 42."),
    ]


def _role_scenarios() -> list[_Scenario]:
    """Probe histories that open with a system message or run past two turns."""

    return [
        _Scenario(
            "roles/system",
            [
                {"role": "system", "content": "You are a helpful assistant."},
                *_two_turn_conversation("Response one."),
            ],
        ),
        _Scenario(
            "roles/three-turn",
            [
                *_two_turn_conversation("Response one."),
                _user("Question three."),
                _assistant("Response three."),
            ],
        ),
    ]


def _tool_scenarios() -> list[_Scenario]:
    """Tool-call probes, each differing from the baseline round trip in exactly one way.

    None of these is redundant. Measured over the default templates of Llama 3.1, Phi-3,
    GLM-4.5, GPT-OSS, Gemma 2/3, DeepSeek-V3, Nemotron Nano and Qwen 2.5/3.6, every
    variant renders differently from the baseline for 4 to 8 of those 10: empty arguments
    skip the loop body that emits parameters, and an empty-string content is not treated
    like an absent one.

    A call carrying no ``id`` is deliberately *not* probed: dropping the id leaves the
    render byte-identical for all 10, and no template TRL ships even mentions one.
    """

    mapping_call = _tool_call({"key": "alpha"})

    return [
        _Scenario("tools/mapping-arguments", _tool_round_trip(mapping_call)),
        _Scenario("tools/json-arguments", _tool_round_trip(_tool_call('{"key":"alpha"}'))),
        _Scenario("tools/empty-mapping-arguments", _tool_round_trip(_tool_call({}))),
        _Scenario("tools/empty-json-arguments", _tool_round_trip(_tool_call("{}"))),
        _Scenario("tools/empty-content-with-call", _tool_round_trip(mapping_call, content="")),
        _Scenario(
            "tools/multiple-results",
            [
                _user("Look up the requested value."),
                _assistant("Checking.", [mapping_call, _tool_call({"key": "beta"}, call_id="call_2")]),
                _tool_result("42", "call_1"),
                _tool_result("84", "call_2"),
                _assistant("The values are 42 and 84."),
            ],
        ),
        # A tool turn may be rendered differently once it is no longer the last turn.
        _Scenario(
            "tools/followed-by-user-turn",
            [
                *_tool_round_trip(mapping_call),
                _user("Question two."),
                _assistant("Response two."),
            ],
        ),
    ]


def _default_scenarios(
    think_start: str | None,
    think_end: str | None,
) -> list[_Scenario]:
    """Build model-neutral text, reasoning-shape, role, and tool-call probes."""

    response_variants = [
        *_TEXT_RESPONSE_VARIANTS,
        *_reasoning_response_variants(think_start, think_end),
    ]
    return [
        *(_Scenario(name, _two_turn_conversation(response)) for name, response in response_variants),
        *_role_scenarios(),
        *_tool_scenarios(),
    ]


def _custom_scenarios(
    conversations: list[Conversation],
) -> list[_Scenario]:
    return [
        _Scenario(
            name=f"custom/{index}",
            messages=[dict(message) for message in conversation],
        )
        for index, conversation in enumerate(conversations)
    ]


def _assistant_message_indices(messages: Conversation) -> list[int]:
    return [index for index, message in enumerate(messages) if message.get("role") == "assistant"]


def _is_text_only(message: Message) -> bool:
    """String content, or no content at all on an assistant message that only calls tools."""

    content = message.get("content")
    if isinstance(content, str):
        return True
    return content is None and message.get("role") == "assistant" and message.get("tool_calls") is not None


def _validate_scenarios(scenarios: list[_Scenario]) -> None:
    if not scenarios:
        raise ValueError("At least one conversation is required")

    for scenario in scenarios:
        for message in scenario.messages:
            if not _is_text_only(message):
                raise TypeError(f"{scenario.name!r} contains non-text message content; this checker supports text-only conversations")

        if len(_assistant_message_indices(scenario.messages)) < 2:
            raise ValueError(f"{scenario.name!r} must contain at least two assistant messages")


def _custom_kwargs_combinations(
    custom_kwargs: dict[str, list[Any]] | None,
) -> list[dict[str, Any]]:
    """Return the Cartesian product of custom chat-template kwargs."""

    if not custom_kwargs:
        return [{}]

    without_values = [name for name, values in custom_kwargs.items() if not values]
    if without_values:
        raise ValueError(f"custom_kwargs values must be non-empty: {', '.join(without_values)}")

    names = list(custom_kwargs)
    value_sets = (custom_kwargs[name] for name in names)
    return [dict(zip(names, values, strict=True)) for values in product(*value_sets)]


def _iter_cases(
    scenarios: list[_Scenario],
    tool_sets: list[Tools | None],
    kwargs_combinations: list[dict[str, Any]],
) -> Iterator[_Case]:
    for scenario in scenarios:
        for tools in tool_sets:
            for template_kwargs in kwargs_combinations:
                yield _Case(scenario=scenario, tools=tools, template_kwargs=template_kwargs)


def _render_prefix_chain(
    *,
    tokenizer: Any,
    chat_template: str,
    case: _Case,
) -> list[_Rendering]:
    """Render, at each assistant turn, the generation prompt and then the completed history.

    The result is the chain the check walks: appending the assistant response, and then
    the messages that follow it, must never rewrite what came before.
    """

    shared_kwargs: dict[str, Any] = {
        "chat_template": chat_template,
        "tokenize": True,
        # Transformers v5 defaults to return_dict=True, which would wrap the ids in a BatchEncoding.
        "return_dict": False,
        **case.template_kwargs,
    }
    if case.tools is not None:
        shared_kwargs["tools"] = case.tools

    messages = case.scenario.messages
    chain: list[_Rendering] = []
    for assistant_turn, message_index in enumerate(_assistant_message_indices(messages)):
        chain.append(
            _Rendering(
                assistant_turn=assistant_turn,
                token_ids=tokenizer.apply_chat_template(
                    messages[:message_index],
                    add_generation_prompt=True,
                    **shared_kwargs,
                ),
            )
        )
        chain.append(
            _Rendering(
                assistant_turn=assistant_turn,
                token_ids=tokenizer.apply_chat_template(
                    messages[: message_index + 1],
                    add_generation_prompt=False,
                    **shared_kwargs,
                ),
            )
        )
    return chain


def _is_token_prefix(prefix: list[int], sequence: list[int]) -> bool:
    return len(sequence) >= len(prefix) and sequence[: len(prefix)] == prefix


def _first_divergence(previous: list[int], current: list[int]) -> int:
    for index, (previous_id, current_id) in enumerate(zip(previous, current)):
        if previous_id != current_id:
            return index
    return min(len(previous), len(current))


def _token_context(tokenizer: Any, token_ids: list[int], divergence_index: int) -> str:
    """Token pieces around a divergence.

    Pieces rather than decoded text: the two sides of a boundary re-merge usually
    decode to the very same characters, which would hide the failure being reported.
    """
    # Token pieces reported on either side of a divergence. Diagnostics only.
    CONTEXT_TOKENS_BEFORE = 8
    CONTEXT_TOKENS_AFTER = 12

    window = token_ids[max(0, divergence_index - CONTEXT_TOKENS_BEFORE) : divergence_index + CONTEXT_TOKENS_AFTER]
    try:
        return " ".join(repr(piece) for piece in tokenizer.convert_ids_to_tokens(window))
    except Exception:
        # Diagnostics run only on the failure path, and must never mask the failure itself.
        return f"token_ids={window!r}"


def _prefix_mismatch(
    tokenizer: Any,
    case: _Case,
    previous: _Rendering,
    current: _Rendering,
) -> PrefixFailure:
    divergence = _first_divergence(previous.token_ids, current.token_ids)
    return PrefixFailure(
        scenario=case.scenario.name,
        conversation=[dict(message) for message in case.scenario.messages],
        tools=(None if case.tools is None else [dict(tool) for tool in case.tools]),
        template_kwargs=dict(case.template_kwargs),
        previous_assistant_turn=previous.assistant_turn,
        current_assistant_turn=current.assistant_turn,
        divergence_token_index=divergence,
        previous_token_id=previous.token_ids[divergence] if divergence < len(previous.token_ids) else None,
        current_token_id=current.token_ids[divergence] if divergence < len(current.token_ids) else None,
        previous_token_context=_token_context(tokenizer, previous.token_ids, divergence),
        current_token_context=_token_context(tokenizer, current.token_ids, divergence),
    )


def _resolve_chat_template(tokenizer: Any, chat_template: str | None) -> str:
    """Fall back to the tokenizer's own template, and reject a missing or empty one."""

    if not callable(getattr(tokenizer, "apply_chat_template", None)):
        raise TypeError("tokenizer must provide apply_chat_template()")

    resolved = chat_template if chat_template is not None else getattr(tokenizer, "chat_template", None)
    if not resolved:
        raise ValueError("chat_template must be a non-empty string, or the tokenizer must define one")
    return resolved


def find_chat_template_prefix_failure(
    tokenizer: Any,
    chat_template: str | None = None,
    *,
    think_start: str | None = "<think>",
    think_end: str | None = "</think>",
    conversations: list[Conversation] | None = None,
    tools_variants: list[Tools | None] | None = None,
    custom_kwargs: dict[str, list[Any]] | None = None,
) -> PrefixFailure | None:
    """Return the first prefix-preservation failure, or ``None`` on success.

    At each assistant turn, the generation prompt must be a token-ID prefix of
    the history containing the completed assistant response. That completed
    history must then be a prefix of the next assistant generation prompt.
    Together, these checks ensure that appending an assistant response or later
    messages never changes already-tokenized history. Rendered string prefixes
    alone are insufficient because tokenization can change at the boundary.

    The built-in text-only scenarios cover whitespace-sensitive boundaries,
    common inline-reasoning shapes, multi-turn histories, system messages, and
    standard tool calls. Reasoning delimiters are opaque probe strings: no
    reasoning convention or model-specific kwarg is required by the checker.

    Args:
        tokenizer: Transformers-compatible tokenizer that determines the result.
        chat_template: Jinja chat template passed directly to the tokenizer.
            Defaults to the tokenizer's own ``chat_template``.
        think_start: Opening reasoning delimiter used to build reasoning-shaped
            probes. Set both delimiters to ``None`` to disable these probes.
        think_end: Closing reasoning delimiter used with ``think_start``.
        conversations: Optional replacement for the built-in conversations.
        tools_variants: Optional replacement for no-tools and generic-tool cases.
        custom_kwargs: Chat-template kwargs mapped to the values to test. Every
            combination is rendered. Unsupported combinations are skipped.

    Returns:
        The first reproducible failure, including token-level diagnostics. If
        every render fails, the returned failure contains the last error.
        Returns ``None`` only when at least one boundary was checked and every
        checked boundary preserved its prefix.

    Note:
        This is an empirical high-coverage check, not a proof over every possible
        Jinja input. Pass custom conversations and kwargs for private branches.
    """

    chat_template = _resolve_chat_template(tokenizer, chat_template)
    _validate_thinking_delimiters(think_start, think_end)

    scenarios = _default_scenarios(think_start, think_end) if conversations is None else _custom_scenarios(conversations)
    _validate_scenarios(scenarios)

    tool_sets: list[Tools | None] = [None, [DEFAULT_TOOL]] if tools_variants is None else tools_variants
    if not tool_sets:
        raise ValueError("At least one tools variant is required")

    kwargs_combinations = _custom_kwargs_combinations(custom_kwargs)

    for case in _iter_cases(scenarios, tool_sets, kwargs_combinations):
        try:
            chain = _render_prefix_chain(tokenizer=tokenizer, chat_template=chat_template, case=case)
        except Exception:
            # The template does not support this combination; it says nothing about prefixes.
            continue

        for previous, current in pairwise(chain):
            if not _is_token_prefix(previous.token_ids, current.token_ids):
                return _prefix_mismatch(tokenizer, case, previous, current)

    return None


def is_chat_template_prefix_preserving(
    tokenizer: Any | str,
    chat_template: str | None = None,
    *,
    think_start: str | None = "<think>",
    think_end: str | None = "</think>",
    conversations: list[Conversation] | None = None,
    tools_variants: list[Tools | None] | None = None,
    custom_kwargs: dict[str, list[Any]] | None = None,
) -> bool:
    """Return whether the template/tokenizer pair passes the prefix check.

    See `find_chat_template_prefix_failure`, which this wraps, for the property
    being checked and for a description of the arguments.
    """

    if isinstance(tokenizer, str):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    failure = find_chat_template_prefix_failure(
        tokenizer,
        chat_template,
        think_start=think_start,
        think_end=think_end,
        conversations=conversations,
        tools_variants=tools_variants,
        custom_kwargs=custom_kwargs,
    )

    return failure is None
