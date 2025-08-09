import math_verify as mv

from src.utils import text as text_utils
from src.configs.markers import Markers
from src.dac_agent import ChatMessage

from experiments.BigCodeBench.Server.code_client import CodeClient, ExecutionResult


def answer_reward(last_message:ChatMessage, result:ExecutionResult) -> float:
    role = last_message.role
    content = last_message.content

    if role != "assistant":
        raise ValueError(f"Expected role 'assistant', got '{role}'")

    # answer_list = text_utils.extract_between(content, Markers.ANSWER_START, Markers.ANSWER_END)

    # if len(answer_list) == 0:
    #     return 0.0

    if result.success:
        return 3.0 * (result.returncode / 100)  # Scale return code to [0, 3]
    elif result.error_type == "timeout":
        return 0.0
    else:
        return 0.0
