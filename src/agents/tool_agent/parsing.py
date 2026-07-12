from __future__ import annotations
from dataclasses import dataclass

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.engine.protocol import ToolCall
from vllm.reasoning import ReasoningParserManager
from vllm.tool_parsers import ToolParserManager
from vllm.tokenizers import TokenizerLike

from src.utils.logging import create_logger

logger = create_logger(__name__)


# The stop tag for each supported parser
STOP_TAGS = {
    "qwen3_xml": "</tool_call>",
    "qwen3_coder": "</tool_call>",
    "step3p5": "</tool_call>",
    "seed_oss": "</seed:tool_call>",
    "glm45": "</tool_call>",
    "glm47": "</tool_call>",
}

# Escape-free (tag-delimited) tool parsers we support
SUPPORTED_PARSERS = set(STOP_TAGS.keys())


@dataclass
class ParsedTurn:
    """The client-side parse of one assistant generation.

    A turn is split into:

    * `reasoning`  — the model's internal `<think>` block (via the reasoning parser);
    * `tool_call`  — the single delegation call, if any (a native vLLM `ToolCall`);
    * `content`    — the remaining text;
    """

    reasoning: str | None
    content: str | None  # TODO: never parsed to None, even though should be sometimes
    tool_call: ToolCall | None = None

    @property
    def has_tool_call(self) -> bool:
        return self.tool_call is not None


class NativeToolParser:
    """Splits an assistant generation into reasoning / content / a single tool call using
    vLLM's native tool and reasoning parsers for the escape-free format `name`."""

    def __init__(
        self,
        name: str,
        tokenizer: TokenizerLike,
        reasoning_parser: str | None = None,
    ) -> None:
        self._name = name
        self._tokenizer = tokenizer
        self.stop_tag = STOP_TAGS[name]

        tool_cls = ToolParserManager.get_tool_parser(name)
        self._tool_parser = tool_cls(tokenizer)

        self._reasoning_parser = None
        if reasoning_parser is not None:
            self._reasoning_parser = ReasoningParserManager.get_reasoning_parser(reasoning_parser)(tokenizer)

    def parse(self, content: str | None) -> ParsedTurn:
        """
        Split reasoning, then extract the single tool call (if any).

        Args:
            content (str | None): the assistant generation to parse.

        Returns:
            ParsedTurn: the parsed reasoning, content, and tool call.
        """
        if content is None:  # No text to parse, return empty turn
            return ParsedTurn(reasoning=None, content=None, tool_call=None)

        # Minimal request object; vLLM's only need its presence.
        dummy_request = ChatCompletionRequest(messages=[], model="_", seed=None)

        reasoning: str | None = None
        if self._reasoning_parser is not None:  # Extract reasoning first
            reasoning, content = self._reasoning_parser.extract_reasoning(content, dummy_request)

        if content is None:  # If no content left, then there is no tool call either
            return ParsedTurn(reasoning=reasoning, content=None, tool_call=None)

        tools_info = self._tool_parser.extract_tool_calls(content, dummy_request)
        assert len(tools_info.tool_calls or []) <= 1, "expected at most one tool call per turn"
        tool_call = tools_info.tool_calls[0] if tools_info.tools_called else None
        return ParsedTurn(reasoning=reasoning, content=content, tool_call=tool_call)


def build_tool_parser(
    name: str,
    tokenizer: TokenizerLike,
    reasoning_parser: str | None = None,
) -> NativeToolParser:
    """Build the parser for the escape-free tool format `name`.

    Args:
        name (str): `multi_turn.format` / vLLM tool-parser name.
        tokenizer (TokenizerLike): the model tokenizer.
        reasoning_parser (str | None): optional vLLM reasoning-parser name.

    Raises:
        ValueError: if `name` is not an escape-free format (JSON/escaping formats such
            as ``"hermes"`` are intentionally unsupported).
    """
    if name not in SUPPORTED_PARSERS:
        raise ValueError(
            f"{name!r} is not a supported tool format. "
            f"Supported: {sorted(SUPPORTED_PARSERS)}. "
            "JSON/escaping formats (hermes, openai, mistral, kimi_k2, pythonic, ...) are intentionally unsupported."
        )
    return NativeToolParser(name, tokenizer, reasoning_parser=reasoning_parser)
