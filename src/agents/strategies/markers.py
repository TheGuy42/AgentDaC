import re


class Markers:
    TASK_START = "<task>"
    TASK_END = "</task>"
    ANS_START = "<answer>"
    ANS_END = "</answer>"
    CLARIFY_START = "<clarify>"
    CLARIFY_END = "</clarify>"
    TOOL_START = "<tool_call>"
    TOOL_END = "</tool_call>"
    THINK_START = "<think>"
    THINK_END = "</think>"


def extract_between(text: str, start_marker: str, end_marker: str, strict: bool = True) -> list[str]:
    """
    Extracts all instances of text between two specific markers in a string.

    - If `strict` is True: Matches text strictly between the first occurrence of `start_marker` and the next occurrence of `end_marker`.
    - If `strict` is False: Matches text starting from each `start_marker` up to the next `end_marker`, next `start_marker`, or end of the string.

    Args:
        text: The input string to parse.
        start_marker: The beginning marker.
        end_marker: The ending marker.
        strict: Whether to use strict matching mode.

    Returns:
        A list of strings, where each string is an instance of text found between the markers.
    """
    s = re.escape(start_marker)
    e = re.escape(end_marker)

    if strict:
        # Between start and end, match absolutely any characters (including newlines).
        pattern = rf"{s}([\s\S]*?){e}"
    else:
        # Start at each start_marker, then capture until the next end/start or end of text.
        # Lookahead ensures we *stop before* whichever comes first, without consuming it.
        pattern = rf"{s}([\s\S]*?)(?={e}|{s}|$)"

    return re.findall(pattern, text)


# TODO: try with strict=False
def extract_answer(text: str, strict=True) -> str:
    answer_list = extract_between(text, Markers.ANS_START, Markers.ANS_END, strict=strict)
    answer = answer_list[-1] if len(answer_list) > 0 else text
    return answer.strip()


# TODO: try with strict=false
def extract_tasks(text: str, strict=True) -> list[str]:
    task_list = extract_between(text, Markers.TASK_START, Markers.TASK_END, strict=strict)
    return [task.strip() for task in task_list]