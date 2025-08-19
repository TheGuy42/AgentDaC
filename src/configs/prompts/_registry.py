import textwrap
from src.utils.logging import create_logger


logger = create_logger(__name__)


PROMPTS: dict[str, str] = {}


def available_prompts() -> list[str]:
    """
    Returns a list of available prompt names.
    """
    return list(PROMPTS.keys())


def add_prompt(name: str, content: str, allow_override: bool = False):
    """
    Add a prompt to the global prompts dictionary.
    """
    if name in PROMPTS:
        if allow_override:
            logger.warning(f"Prompt '{name}' already exists. Overring with this prompt.")
        else:
            raise ValueError(f"Prompt '{name}' already exists.")

    PROMPTS[name] = textwrap.dedent(content).strip()


def get_prompt(name: str | None) -> str | None:
    """
    Get a prompt by its name.
    - If the name is None or empty, returns `None`.
    - Else, if the prompt does not exist, raises a `ValueError`.
    - Otherwise, returns the prompt content.
    """
    if name is None or name == "":
        return None

    if name not in PROMPTS:
        raise ValueError(f"Prompt '{name}' does not exist. Available prompts: {available_prompts()}")

    return PROMPTS[name]
