from art import Trajectory
import re


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


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def print_trajectory(trajectory: Trajectory):
    for message in trajectory.messages():
        role = f"{bcolors.OKGREEN}{message['role']}{bcolors.ENDC}:: "
        content = message["content"]
        content = content.replace("<task>", f"{bcolors.OKCYAN}<task>{bcolors.ENDC}")
        content = content.replace("</task>", f"{bcolors.OKCYAN}</task>{bcolors.ENDC}")
        content = content.replace("<answer>", f"{bcolors.OKBLUE}<answer>{bcolors.ENDC}")
        content = content.replace("</answer>", f"{bcolors.OKBLUE}</answer>{bcolors.ENDC}")

        print(f"{role}{content}")
