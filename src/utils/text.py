import re

from src.configs.markers import Markers


def extract_text_between_markers(text: str, start_marker: str, end_marker: str) -> list[str]:
    """
    Extracts all instances of text between two specific markers in a string.

    Args:
        text: The input string to parse.
        start_marker: The beginning marker.
        end_marker: The ending marker.

    Returns:
        A list of strings, where each string is an instance of text found between the markers.
    """
    # Create a regex pattern to find text non-greedily between the markers
    # re.escape is used to escape any special characters in the markers
    # (.*?) matches any character (except newline) zero or more times, non-greedily
    pattern = re.escape(start_marker) + r"(.*?)" + re.escape(end_marker)

    # Find all non-overlapping matches of the pattern in the string
    matches = re.findall(pattern, text, re.DOTALL)

    return matches


def extract_answer(text: str) -> str:
    answer_list = extract_text_between_markers(text, Markers.ANSWER_START, Markers.ANSWER_END)
    if len(answer_list) > 0:
        answer = answer_list[-1]  # Take the last answer if multiple are found
    else:
        answer = text
    return answer.strip()


def extract_tasks(text: str) -> list[str]:
    task_list = extract_text_between_markers(text, Markers.TASK_START, Markers.TASK_END)
    return [task.strip() for task in task_list]
