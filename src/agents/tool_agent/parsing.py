from __future__ import annotations
from dataclasses import dataclass

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.engine.protocol import ToolCall
from vllm.reasoning import ReasoningParserManager
from vllm.tool_parsers import ToolParserManager
from vllm.tokenizers import TokenizerLike, get_tokenizer

from src.inference import InferenceResponse
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
    content: str | None
    tool_call: ToolCall | None = None

    @property
    def has_tool_call(self) -> bool:
        return self.tool_call is not None


class NativeParser:
    """Splits an assistant generation into reasoning / content / a single tool call using
    vLLM's native tool and reasoning parsers for the escape-free format `name`."""

    def __init__(
        self,
        tool_parser: str,
        tokenizer: TokenizerLike,
        reasoning_parser: str | None = None,
    ) -> None:

        self.stop_tag = STOP_TAGS[tool_parser]

        tool_cls = ToolParserManager.get_tool_parser(tool_parser)
        self._tool_parser = tool_cls(tokenizer)

        self._reasoning_parser = None
        if reasoning_parser is not None:
            reasoning_cls = ReasoningParserManager.get_reasoning_parser(reasoning_parser)
            self._reasoning_parser = reasoning_cls(tokenizer)

    def parse(self, response: InferenceResponse) -> ParsedTurn:
        """
        Split reasoning, then extract the single tool call (if any).

        Args:
            response (InferenceResponse): the assistant generation to parse.

        Returns:
            ParsedTurn: the parsed reasoning, content, and tool call.
        """
        if response.tool_calls and len(response.tool_calls) > 1:
            logger.warning("Expected at most one tool call per turn, but got multiple.")

        # Minimal request object; vLLM's only need its presence.
        dummy_request = ChatCompletionRequest(messages=[], model="_", seed=None)

        reasoning = response.reasoning
        content = response.content
        tool_call = response.tool_calls[0] if response.tool_calls else None

        # Try to extract reasoning if missing
        if (self._reasoning_parser is not None) and (content is not None) and (reasoning is None):
            reasoning, content = self._reasoning_parser.extract_reasoning(content, dummy_request)

        # Try to extract tool calls if missing
        if (content is not None) and (tool_call is None):
            tools_info = self._tool_parser.extract_tool_calls(content, dummy_request)
            if tools_info.tools_called and len(tools_info.tool_calls) > 1:
                logger.warning("Expected at most one tool call per turn, but parsed multiple.")

            content = tools_info.content
            tool_call = tools_info.tool_calls[0] if tools_info.tool_calls else None

        return ParsedTurn(reasoning=reasoning, content=content, tool_call=tool_call)


def build_tool_parser(
    tool_parser: str,
    tokenizer: TokenizerLike | str,
    reasoning_parser: str | None = None,
    **kwargs,
) -> NativeParser:
    """Build the parser for the escape-free tool format `tool_parser`.

    Args:
        tool_parser (str): `multi_turn.format` / vLLM tool-parser name.
        tokenizer (TokenizerLike | str): tokenizer or tokenizer name.
        reasoning_parser (str | None): optional vLLM reasoning-parser name.
        **kwargs: additional keyword arguments for the tokenizer initialization (if `tokenizer` is a string).

    Raises:
        ValueError: if `tool_parser` is not an escape-free format (JSON/escaping formats such
            as ``"hermes"`` are intentionally unsupported).
    """

    if isinstance(tokenizer, str):
        tokenizer = get_tokenizer(tokenizer, **kwargs, tokenizer_cls=TokenizerLike)

    if tool_parser not in SUPPORTED_PARSERS:
        raise ValueError(
            f"{tool_parser!r} is not a supported tool format. "
            f"Supported: {sorted(SUPPORTED_PARSERS)}. "
            "JSON/escaping formats (hermes, openai, ...) are intentionally unsupported."
        )
        
    return NativeParser(tool_parser, tokenizer, reasoning_parser=reasoning_parser)
