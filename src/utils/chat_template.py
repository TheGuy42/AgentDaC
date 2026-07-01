from __future__ import annotations
from transformers import AutoTokenizer
from trl.chat_template_utils import get_training_chat_template, is_chat_template_prefix_preserving
from src.utils.logging import create_logger


logger = create_logger(__name__)


def resolve_chat_template(
    model_path: str,
    manual_template: str | None,
    trust_remote_code: bool = False,
) -> str | None:
    """
    Resolve the chat template to use for training, or raise if none is safe.

    Raises `ValueError` when the template is not prefix-preserving and TRL cannot
    patch it (unknown template).
    """

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    if manual_template is not None:
        tokenizer.chat_template = manual_template

    if tokenizer.chat_template is None:
        raise ValueError(f"Model {model_path} has no chat template, and no manual override was provided.")

    source = "manual custom_chat_template" if manual_template is not None else f"model default {model_path}"

    try:
        patched_template = get_training_chat_template(tokenizer)
        if patched_template is not None:
            logger.info(f"TRL: Patched chat template from {source} to be prefix-preserving for TRL training.")
            tokenizer.chat_template = patched_template

    except ValueError:
        # NOTE: Raised when can't patch, but note that it sometimes tries to patch for other reasons,
        # not only when the template is not prefix-preserving. So we don't raise here, but we do check below.
        pass

    if not is_chat_template_prefix_preserving(tokenizer):
        # NOTE: `is_chat_template_prefix_preserving` is not always accurate,
        # so it should't be relied as a guard to check whether patching is needed or not.
        raise ValueError(f"TRL: Could not patch chat template from {source} to be prefix-preserving. Please provide a manual override.")

    return tokenizer.chat_template
