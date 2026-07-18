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

    def validate(self, content: Any) -> str | None:
        """Return a simple marker-format error, or None if the structure looks valid."""

        if not isinstance(content, str):
            return f"Content must be a string, got {type(content).__name__}."

        marker_pairs = (
            ("task", Markers.TASK_START, Markers.TASK_END),
            ("answer", Markers.ANS_START, Markers.ANS_END),
        )

        all_markers = (
            Markers.TASK_START,
            Markers.TASK_END,
            Markers.ANS_START,
            Markers.ANS_END,
        )

        found_blocks = 0

        for name, start, end in marker_pairs:
            num_starts = content.count(start)
            num_ends = content.count(end)

            if num_starts != num_ends:
                return f"Unbalanced {name} markers: found {num_starts} opening and {num_ends} closing markers."

            blocks = extract_between(content, start, end, strict=True)
            found_blocks += len(blocks)

            # Equal counts but fewer extracted blocks usually means malformed
            # ordering or nesting.
            if len(blocks) != num_starts:
                return f"Malformed or nested {name} markers."

            for block in blocks:
                if not block.strip():
                    return f"Empty {name} block."

                if any(marker in block for marker in all_markers):
                    return f"Nested markers inside a {name} block."

        if found_blocks == 0:
            return "No <task> or <answer> block found."

        return None

    def parse(self, content: Any) -> MarkerTurn:
        if not isinstance(content, str):
            raise ValueError(f"Content to parse must be a string, got {type(content).__name__}")

        if self.strict:
            if error := self.validate(content):
                raise ValueError(error)

        tasks = [t.strip() for t in extract_between(content, Markers.TASK_START, Markers.TASK_END, strict=self.strict)]
        answers = [a.strip() for a in extract_between(content, Markers.ANS_START, Markers.ANS_END, strict=self.strict)]

        if len(tasks) == 0 and len(answers) == 0:
            raise ValueError("Could not parse any <task> or <answer> blocks in the content.")

        return MarkerTurn(
            tasks=tasks if len(tasks) > 0 else None,
            answers=answers if len(answers) > 0 else None,
            raw=content,
        )
