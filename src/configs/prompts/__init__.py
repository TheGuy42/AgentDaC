from src.configs.prompts._registry import (
    available_prompts,
    add_prompt,
    get_prompt,
    PROMPTS,
)

__all__ = [
    "available_prompts",
    "add_prompt",
    "get_prompt",
    "PROMPTS",
]

# import modules to register relevant prompts
import src.configs.prompts.gilad_prompts  # noqa: F401
import src.configs.prompts.guided_prompts  # noqa: F401
import src.configs.prompts.guy_prompts  # noqa: F401
import src.configs.prompts.old_prompts  # noqa: F401
import src.configs.prompts.regex_prompts  # noqa: F401
import src.configs.prompts.tasks_depleted  # noqa: F401
import src.configs.prompts.tool_prompts  # noqa: F401
