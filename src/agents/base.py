from __future__ import annotations
from abc import ABC, abstractmethod
from enum import StrEnum

from src.inference import InferenceClient, InferenceResponse
from src.trajectory import Trajectory
from src.utils.visualize import trajectory_string, message_string
from src.utils.logging import create_logger
from src.aliases import Message, SystemMessage
from src.configs import PromptConfig, DecompConfig


logger = create_logger(__name__)


class BaseAgent(ABC):
    @classmethod
    @abstractmethod
    def error_kinds(cls) -> type[StrEnum]:
        """
        Get the enumeration of error kinds for this agent.

        Returns:
            type[StrEnum]: The enumeration of error kinds.
        """
        pass

    def __init__(
        self,
        client: InferenceClient,
        prompt_config: PromptConfig,
        decomp_config: DecompConfig,
        current_depth: int = 0,
        additional_histories: bool = False,
        verbose: bool = False,
    ):
        self.client = client
        self.prompt_config = prompt_config.initialize()
        self.decomp_config = decomp_config.clone()
        self.current_depth = current_depth
        self.additional_histories = additional_histories
        self.verbose = verbose

        self.trajectory = Trajectory(messages_and_responses=[])

        if sys_msg := self._system_prompt():
            self.append_message(sys_msg)

    @property
    def metrics(self) -> dict[str, float | int | bool]:
        return self.trajectory.metrics

    @property
    def metadata(self) -> dict[str, float | int | str | bool | None]:
        return self.trajectory.metadata

    def __str__(self) -> str:
        return trajectory_string(self.trajectory)

    def _system_prompt(self) -> SystemMessage | None:
        if self.current_depth == 0:
            content = self.prompt_config.system_root
        elif self.current_depth < self.decomp_config.max_depth:
            content = self.prompt_config.system_inter
        else:
            content = self.prompt_config.system_leaf

        if content is not None:
            return SystemMessage(role="system", content=content)

        return None

    def append_message(self, message: Message | InferenceResponse):
        """
        Append a message to the trajectory and print it if verbose is enabled.

        Args:
            message (Message | InferenceResponse): The message to append.
        """
        self.trajectory.messages_and_responses.append(message)
        if self.verbose:
            if isinstance(message, InferenceResponse):
                message = self.trajectory.messages()[-1]
            print(message_string(message, indent=self.current_depth))

    async def call(self, messages: list[Message], **kwargs) -> InferenceResponse:
        """
        Generate an assistant response via the inference client.
        Should not be used directly; use `chat` instead.

        Args:
            messages (list[Message]): The list of messages to send.
            **kwargs: Additional keyword arguments forwarded to the client.
        """
        return await self.client.chat(messages, **kwargs)

    async def answer(self, prompt: Message, **kwargs) -> str | None:
        """
        Answer a question using the agent.

        Args:
            prompt (Message): The question to answer.
            **kwargs: Additional keyword arguments to pass to OpenAI API call.
        Returns:
            (str | None): The answer text from the agent, or None if the answer cannot be parsed.
        """
        trajectory = await self.chat(prompt, **kwargs)
        answer = self.parse_answer(trajectory.messages_and_responses[-1])
        return answer.strip() if isinstance(answer, str) else answer

    @abstractmethod
    async def chat(self, prompt: Message, **kwargs) -> Trajectory:
        """
        Start a conversation with the agent using the provided prompt.

        Args:
            prompt (Message): The initial message to start the conversation.
            **kwargs: Additional keyword arguments to pass to OpenAI API call.

        Returns:
            (Trajectory): The trajectory of the conversation, including messages and responses.
        """
        pass

    @abstractmethod
    def parse_answer(self, message: Message | InferenceResponse) -> str | None:
        """
        Parse the final answer from the agent's message.

        Args:
            message (Message | InferenceResponse): The agent's message containing the answer.

        Returns:
            (str | None): The parsed answer text, or None if the answer cannot be parsed.
        """
        pass
