import math_verify as mv
from math_verify.errors import TimeoutException

import src.agents.marker_agent.markers as markers
from src.openai_types import Message
from src.utils.logging import create_logger


logger = create_logger(__name__)


def answer_reward(sample: dict[str, str], message: Message) -> tuple[float, bool]:
    """
    Answer correctness reward function.

    Args:
        sample (dict): A dictionary containing all relevant ground truth information.
        message (Message): The message object containing the model's response.

    Returns:
        (tuple[float, bool]): A tuple (reward, parsed) where reward is 1.0 if the answer is correct, 0.0 otherwise,
            and parsed is True if the answer was successfully parsed, False otherwise.
    """
    try:
        content = message.get("content")
        assert message["role"] == "assistant", f"Expected role 'assistant', got '{message['role']}'"
        assert isinstance(content, str), f"Expected content to be a string, got {type(content)}"

        gold_answer = sample["answer"]
        llm_answer = markers.extract_answer(content)

        gold_parsed = mv.parse(f"${gold_answer}$", raise_on_error=True)
        llm_parsed = mv.parse(llm_answer, raise_on_error=True)
        is_correct = mv.verify(gold_parsed, llm_parsed, raise_on_error=True)
        return (1.0 if is_correct else 0.0, True)

    except TimeoutException as e:
        logger.info(f"Timeout during answer reward computation: {e}")
        return (0.0, False)
    except Exception as e:
        logger.warning(f"Error during answer reward computation: {e}")
        return (0.0, False)
