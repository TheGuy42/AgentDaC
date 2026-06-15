from src.utils.logging import create_logger


logger = create_logger(__name__)


def answer_reward(sample: dict, model_answer: str) -> tuple[float, bool]:
    """
    Answer correctness reward function for the memorization task.

    The task has no semantics to verify: each sample carries a hidden label drawn from
    `sample["labels"]` and the reward is simply whether the model reproduced it.
    """
    valid_answers = {label.strip().upper() for label in sample["labels"]}
    gold_answer = sample["answer"].strip().upper()
    pred_answer = model_answer.strip().upper()

    parse_success = pred_answer in valid_answers
    is_correct = parse_success and pred_answer == gold_answer
    return (1.0 if is_correct else 0.0, parse_success)
