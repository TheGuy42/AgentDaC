from art import Trajectory
from art.types import Message
from src.configs.markers import Markers
import re


class Colors:
    """ANSI color codes for terminal output."""

    GREEN = "\033[92m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def colorize_markers(content: str) -> str:
    """Add color formatting to special markers in content."""
    marker_colors = {
        Markers.TASK_START: Colors.CYAN,
        Markers.TASK_END: Colors.CYAN,
        Markers.ANSWER_START: Colors.BLUE,
        Markers.ANSWER_END: Colors.BLUE,
    }

    for marker, color in marker_colors.items():
        colored = f"{color}{marker}{Colors.RESET}"
        content = content.replace(marker, colored)

    return content


def normalize_newlines(text: str) -> str:
    """Normalize all newline types to standard \\n."""
    return re.sub(r"\r\n|\r|\n|\u000B|\u000C|\u0085|\u2028|\u2029", "\n", text)


def message_string(message: Message, indent: int = 0) -> str:
    """
    Format a message with proper indentation.

    Output format:
        **role:** [role text]
        **content:** [content text line 1]
        [content text line 2] (if newlines exist in content)

    Args:
        message: The message to format
        indent: Number of indentation levels (each level = 2 spaces)

    Returns:
        Formatted message string with proper indentation
    """
    spaces = "    " * indent

    # Format role line with bold formatting
    bold_role_label = f"{Colors.BOLD}Role:{Colors.RESET}"
    role_line = f"{spaces}{bold_role_label} {Colors.GREEN}{message['role']}{Colors.RESET}"

    # Format content with colors and proper indentation
    content = colorize_markers(message.get("content", ""))
    bold_content_label = f"{Colors.BOLD}Content:{Colors.RESET}"
    content_first_line = f"{spaces}{bold_content_label} "

    # Handle multi-line content properly
    lines = normalize_newlines(content).split("\n")
    content_lines = []

    for i, line in enumerate(lines):
        if i == 0:
            # First line gets the **content:** label
            content_lines.append(content_first_line + line)
        else:
            # Subsequent lines get proper indentation
            content_lines.append(spaces + line)

    indented_content = "\n".join(content_lines)

    return f"{role_line}\n{indented_content}\n"


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
