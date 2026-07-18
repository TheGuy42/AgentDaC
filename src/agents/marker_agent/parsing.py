from __future__ import annotations
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from src.agents.marker_agent.markers import Markers, extract_between


class MarkerAction(StrEnum):
    """The marker protocol's actions. Used by the agent to express policy; the parser
    itself is action-agnostic."""

    TASK = "task"
    ANSWER = "answer"


@dataclass
class MarkerTurn:
    """The blocks extracted from a single assistant turn.

    A field is `None` when that block kind is absent, and a non-empty list of payloads
    when present (normally a single element, since the default stop tags bound a turn to
    one block).
    """

    tasks: list[str] | None
    answers: list[str] | None
    raw: str


class MarkerParser:
    """Extracts the marker blocks of a turn.

    The parser is a pure extractor: it knows nothing about which actions are allowed and
    never raises. The agent inspects the result (which fields are `None`), detects
    malformed turns, and enforces the allowed-action policy.

    Args:
        strict (bool):
            - True: requires balanced (closed) tags (an unclosed block yields nothing).
            - False: tolerant — captures from each start marker to the next tag or the
              end of the string.
    """

    def __init__(self, strict: bool = True) -> None:
        self.strict = strict

    def stop_tags(self) -> list[str]:
        """End markers used as generation stop strings (both block kinds)."""
        return [Markers.TASK_END, Markers.ANS_END]

    def parse(self, content: Any) -> MarkerTurn:
        if not isinstance(content, str):
            raise ValueError(f"Content to parse must be a string, got {type(content).__name__}")

        tasks = [t.strip() for t in extract_between(content, Markers.TASK_START, Markers.TASK_END, strict=self.strict)]
        answers = [a.strip() for a in extract_between(content, Markers.ANS_START, Markers.ANS_END, strict=self.strict)]

        return MarkerTurn(tasks=tasks or None, answers=answers or None, raw=content)
