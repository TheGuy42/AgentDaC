from src.openai_types import Message
from experiments.big_code_bench.server.code_client import CodeClient, ExecutionResult


def answer_reward(message: Message, result: ExecutionResult) -> float:
    content = message.get("content")
    assert message["role"] == "assistant", f"Expected role 'assistant', got '{message['role']}'"
    assert isinstance(content, str), f"Expected content to be a string, got {type(content)}"

    # answer_list = text_utils.extract_between(content, Markers.ANSWER_START, Markers.ANSWER_END)

    # if len(answer_list) == 0:
    #     return 0.0

    if result.success:
        return 3.0 * (result.returncode / 100)  # Scale return code to [0, 3]
    elif result.error_type == "timeout":
        return 0.0
    else:
        return 0.0
