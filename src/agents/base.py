from __future__ import annotations
from abc import ABC, abstractmethod

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
from art import Trajectory

from src.utils.visualize import trajectory_string
from src.utils.logging import create_logger
from src.openai_types import Message, SystemMessage
from src.configs import PromptConfig, DecompConfig
from src.configs.prompts import get_prompt


logger = create_logger(__name__)


class BaseAgent(ABC):
    def __init__(
        self,
        openai_client: AsyncOpenAI,
        model_name: str,
        prompt_config: PromptConfig,
        decomp_config: DecompConfig,
        current_depth: int = 0,
        additional_histories: bool = False,
    ):
        self.openai_client = openai_client
        self.model = model_name
        self.prompt_config = prompt_config
        self.decomp_config = decomp_config.clone()
        self.current_depth = current_depth
        self.additional_histories = additional_histories

        self.trajectory = Trajectory(
            messages_and_choices=[],
            additional_histories=[],
            reward=0,
            metrics={
                "direct_calls": 0,
                "total_calls": 0,
                "direct_tasks": 0,
                "total_tasks": 0,
                "max_depth": 0,
            },
        )

        if sys_msg := self._get_system_message():
            self.trajectory.messages_and_choices.append(sys_msg)

    @property
    def metrics(self) -> dict[str, float | int | bool]:
        return self.trajectory.metrics

    @property
    def metadata(self) -> dict[str, float | int | str | bool | None]:
        return self.trajectory.metadata

    def __str__(self) -> str:
        return trajectory_string(self.trajectory)

    def _get_system_message(self) -> SystemMessage | None:
        pc = self.prompt_config
        dc = self.decomp_config

        if self.current_depth == 0:
            content = get_prompt(pc.system_root)
        elif self.current_depth < dc.max_depth:
            content = get_prompt(pc.system_inter)
        else:
            content = get_prompt(pc.system_leaf)

        if content is not None:
            return SystemMessage(role="system", content=content)

        return None

    async def call(self, messages: list[Message], **kwargs) -> ChatCompletion:
        """
        Call the OpenAI API to get a chat completion.
        Should not be used directly; use `chat` instead.

        Args:
            messages (list[Message]): The list of messages to send to the API.
            **kwargs: Additional keyword arguments to pass to the API call.
        """
        return await self.openai_client.chat.completions.create(
            model=self.model,
            messages=messages,
            logprobs=True,
            **kwargs,
        )

    async def answer(self, prompt: Message, verbose: bool = False, **kwargs) -> str:
        """
        Answer a question using the agent.

        Args:
            prompt (Message): The question to answer.
            verbose (bool): If True, print the conversation messages.
            **kwargs: Additional keyword arguments to pass to OpenAI API call.
        Returns:
            (str): The answer text from the agent.
        """
        trajectory = await self.chat(prompt, verbose=verbose, **kwargs)
        return self.parse_answer(trajectory.messages()[-1]).strip()

    @abstractmethod
    async def chat(
        self,
        prompt: Message,
        verbose: bool = False,
        **kwargs,
    ) -> Trajectory:
        """
        Start a conversation with the agent using the provided prompt.

        Args:
            prompt (Message): The initial message to start the conversation.
            verbose (bool): If True, print the conversation messages.
            **kwargs: Additional keyword arguments to pass to OpenAI API call.

        Returns:
            (Trajectory): The trajectory of the conversation, including messages and choices.
                This trajectory is used to train an `art.TrainableModel` model.
        """
        pass

    @staticmethod
    @abstractmethod
    def parse_answer(message: Message) -> str:
        """
        Parse the final answer from the agent's message.

        Args:
            message (Message): The agent's message containing the answer.

        Returns:
            (str): The parsed answer text.
        """
        pass
