from src.utils.logging import create_logger
import re


logger = create_logger(__name__)


def answer_reward(sample: dict[str, str], model_answer: str) -> tuple[float, bool]:
    """
    Answer correctness reward function.

    Args:
        sample (dict): A dictionary containing all relevant ground truth information.
        model_answer (str): The model's answer as a string.

    Returns:
        (tuple[float, bool]): A tuple (reward, parsed) where reward is 1.0 if the answer is correct, 0.0 otherwise,
            and parsed is True if the answer was successfully parsed, False otherwise.
    """

    try:
        # match (X) pattern
        matches = re.findall(r"\(\s*([A-Za-z])\s*\)", model_answer)
        if len(matches) == 0:
            return (0.0, False)

        last_match = matches[-1]
        if not isinstance(last_match, str):
            logger.warning(f"Expected match to be a string, got {type(last_match)}")
            return (0.0, False)

        gold_answer = sample["answer"]
        is_correct = gold_answer.strip().casefold() == last_match.strip().casefold()
        return (1.0 if is_correct else 0.0, True)

    except Exception as e:
        logger.warning(f"Error during answer reward computation: {e}")
        return (0.0, False)
