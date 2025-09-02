from art import Trajectory
from src.openai_types import Message
import re


class Colors:
    """ANSI color codes for terminal output."""

    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    PURPLE = "\033[95m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    RESET = "\033[0m"


def normalize_newlines(text: str) -> str:
    """Normalize all newline types to standard \\n."""
    return re.sub(r"\r\n|\r|\n|\u000B|\u000C|\u0085|\u2028|\u2029", "\n", text)


def format_field(name: str, lines: list[str], indent: int) -> str:
    """Format a field with a name and multiple lines."""
    if len(lines) == 0:
        return ""

    spaces = (" " * 4) * indent
    fmt_name = f"{spaces}{Colors.BOLD}{name.title()}:{Colors.RESET}"
    fmt_lines = [f"{spaces}{line}" for line in lines]
    return "\n".join([fmt_name] + fmt_lines)


def message_string(message: Message, indent: int = 0) -> str:
    """
    Format a message with proper indentation and colorization.

    Args:
        message: The message to format
        indent: Number of indentation levels (each level = 2 spaces)

    Returns:
        Formatted message string with proper indentation
    """
    texts = []

    role_text = format_field("role", [message["role"].upper()], indent)
    texts.append(role_text)

    if content := message.get("content"):
        if not isinstance(content, str):
            raise ValueError("Message content must be a string.")
        content_lines = normalize_newlines(content).strip().split("\n")
        content_text = format_field("content", content_lines, indent)
        texts.append(content_text)

    if refusal := message.get("refusal"):
        if not isinstance(refusal, str):
            raise ValueError("Message refusal must be a string.")
        refusal_lines = normalize_newlines(refusal).strip().split("\n")
        refusal_text = format_field("refusal", refusal_lines, indent)
        texts.append(refusal_text)

    if tool_calls := message.get("tool_calls"):
        if not isinstance(tool_calls, list):
            raise ValueError("Message tool_calls must be a list.")
        tool_lines = []
        for tool_call in tool_calls:
            fn_name = tool_call["function"]["name"]
            fn_args = tool_call["function"]["arguments"]
            tool_lines.append(f"{fn_name}({fn_args.strip()})")
        tool_text = format_field("tool calls", tool_lines, indent)
        texts.append(tool_text)

    return "\n".join(texts) + "\n"


def trajectory_string(trajectory: Trajectory, indent: int = 0) -> str:
    """
    Format a trajectory with proper indentation.

    Args:
        trajectory: The trajectory to format
        indent: Number of indentation levels for all messages

    Returns:
        Formatted trajectory string
    """
    if not trajectory.messages():
        return ""

    formatted_messages = [message_string(message, indent) for message in trajectory.messages()]

    return "".join(formatted_messages)
