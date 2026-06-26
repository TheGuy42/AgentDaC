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

    source = "manual custom_chat_template" if manual_template is not None else f"model default {model_path}"

    # NOTE: we gate on TRL's `is_chat_template_prefix_preserving` rather than calling
    # `get_training_chat_template` directly, because the latter also requires
    # `{% generation %}` markers and would raise on an already-prefix-preserving template that merely lacks them.
    if is_chat_template_prefix_preserving(tokenizer):
        logger.debug(f"TRL :: Chat template is prefix-preserving ({source}); no patch needed.")
        return manual_template

    # Not prefix-preserving (manual override or model default alike) -> patch via TRL.
    # get_training_chat_template raises ValueError when it cannot patch the template.
    logger.warning(
        f"TRL :: Chat template is not prefix-preserving ({source}); "
        f"patching it to a prefix-preserving variant to avoid corrupting the trajectory token sequence."
    )

    patched = get_training_chat_template(tokenizer)

    tokenizer.chat_template = patched
    if not is_chat_template_prefix_preserving(tokenizer):
        raise ValueError("TRL returned a patched chat template that is still not prefix-preserving. Refusing to proceed.")

    return patched
