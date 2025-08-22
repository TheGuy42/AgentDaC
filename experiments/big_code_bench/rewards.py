from src.openai_types import Message
from src.utils.text import extract_answer
from experiments.big_code_bench.server.code_client import CodeClient, ExecutionResult
from experiments.big_code_bench.format import create_test_code


def execute_code(sample: dict[str, str], message: Message) -> ExecutionResult:
    content = message.get("content")
    assert message["role"] == "assistant", f"Expected role 'assistant', got '{message['role']}'"
    assert isinstance(content, str), f"Expected content to be a string, got {type(content)}"

    agent_answer = extract_answer(content)
    agent_test_code = create_test_code(sample, agent_answer)
    client = CodeClient(port=8002, timeout_buffer=5)
    return client.execute_code(agent_test_code, execution_timeout=60)


def answer_reward(result: ExecutionResult) -> float:
    if result.success:
        return result.returncode / 100
    else:
        return 0.0
