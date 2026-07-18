from __future__ import annotations
import re

from src.agents.perst_agent.perst_agent import PersistentAgent
from src.agents.perst_agent.actions import TurnAction
from src.agents.regex_agent.regex_agent import GuidedRegex
from src.configs import PromptConfig, DecompConfig
from src.aliases import Message
from src.inference import InferenceClient, InferenceResponse
from src.utils.logging import create_logger


logger = create_logger(__name__)


class NativeGuidedRegex(GuidedRegex):
    """
    Same as GuidedRegex, but its parsing allows for arbitrary text
    before the "Action: ..." line, since it may contain the native thinking block of the model.
    """

    def __init__(self, *actions: str) -> None:
        if not actions:
            raise ValueError("At least one allowed action must be provided.")

        if any(a == TurnAction.THINK for a in actions):
            raise ValueError("NativeGuidedRegex does not allow the THINK action.")

        self.actions = actions

        alt = "|".join(re.escape(act) for act in self.actions)
        self.model_pattern = rf"^\s{{0,4}}Action: (?:{alt})\r?\nText: [\s\S]*$"
        self.parse_pattern = rf"[\s\S]*Action: (?P<action>{alt})\r?\nText: (?P<text>[\s\S]*)$"
        self.regex = re.compile(self.parse_pattern)


class NativePersistentAgent(PersistentAgent):
    """
    Same as PersistentAgent, but does not allow the `TurnAction.THINK` action.
    This agent should be used when toggling `enable_thinking=True` and using an explicit `reasoning-parser`.
    In this case, each turn will start from a native thinking block of the model, followed by a guided regex block.

    *Note:* If using this agent without a `reasoning-parser`, then the GuidedRegex will be applied to the entire output
    and will suppress the native thinking block, so a reasoning-parser is required.
    """

    def __init__(
        self,
        client: InferenceClient,
        prompt_config: PromptConfig,
        decomp_config: DecompConfig,
        current_depth: int = 0,
        additional_histories: bool = False,
        verbose: bool = False,
    ):
        super().__init__(
            client=client,
            prompt_config=prompt_config,
            decomp_config=decomp_config,
            current_depth=current_depth,
            additional_histories=additional_histories,
            force_thinking=False,
            verbose=verbose,
        )

    def _create_regex(self) -> GuidedRegex:
        # Same as PersistentAgent, but think action is not allowed.
        regex = super()._create_regex()
        return NativeGuidedRegex(*[a for a in regex.actions if a != TurnAction.THINK])

    def _create_subagent(self) -> NativePersistentAgent:
        return NativePersistentAgent(
            client=self.client,
            prompt_config=self.prompt_config,
            decomp_config=self.decomp_config,
            current_depth=self.current_depth + 1,
            additional_histories=False,  
            verbose=self.verbose,
        )

    def parse_answer(self, message: Message | InferenceResponse) -> str | None:
        if not isinstance(message, InferenceResponse):
            logger.error(f"Expected an InferenceResponse, got {type(message)}")
            return None

        content = message.content
        if not isinstance(content, str):
            logger.error(f"Expected message content to be a string, got {type(content)}")
            return None

        try:
            schema = NativeGuidedRegex(TurnAction.ANSWER)
            turn = schema.parse(content)
            return turn.text
        
        except Exception as e:
            logger.error(f"Failed to parse final answer: {e}")
            return None
