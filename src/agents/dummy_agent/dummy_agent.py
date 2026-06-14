from __future__ import annotations
from src.trajectory import Trajectory
from src.agents.base import BaseAgent
from src.aliases import Message, UserMessage, Response
from src.utils.visualize import trajectory_string, message_string
from src.utils.logging import create_logger


logger = create_logger(__name__)


class DummyAgent(BaseAgent):
    """
    A simple model wrapper that satisfies the BaseAgent interface.
    Does not create any sub-agents or any custom logic.
    """

    def __init__(
        self,
        openai_client,
        model_name,
        prompt_config,
        decomp_config,
        **kwargs,
    ):

        super().__init__(
            openai_client,
            model_name,
            prompt_config,
            decomp_config,
            current_depth=0,
            additional_histories=False,
        )

        if kwargs:
            logger.warning(f"DummyAgent ignores additional kwargs: {kwargs}")

    async def call(self, messages: list[Message], **kwargs) -> Response:
        extra_body: dict = kwargs.setdefault("extra_body", {})
        extra_body.setdefault("include_stop_str_in_output", True)
        return await super().call(messages, **kwargs)

    async def chat(
        self,
        prompt: Message,
        verbose: bool = False,
        **kwargs,
    ) -> Trajectory:
        if prompt.get("role") != "user":
            logger.warning(f"Prompt role is expected to be 'user', but got {prompt.get('role')}.")

        self.decomp_config.reset()
        self.trajectory.messages_and_responses.append(prompt)

        if verbose:
            print(trajectory_string(self.trajectory, indent=self.current_depth))

        self.metrics.setdefault("direct_thinks", 0)
        self.metrics.setdefault("total_thinks", 0)

        # Model turn
        completion = await self.call(self.trajectory.messages(), **kwargs)
        self.trajectory.messages_and_responses.append(completion)

        # Update metrics
        self.metrics["total_calls"] += 1
        self.metrics["direct_calls"] += 1
        if completion.usage is not None:
            self.metrics["direct_tokens"] = completion.usage.total_tokens

        if verbose:
            print(message_string(self.trajectory.messages()[-1], indent=self.current_depth))

        self.decomp_config.update_round(num_tasks=0)

        # Update final stats
        self.metrics["response_completed"] = completion.choices[0].finish_reason != "length"
        self.trajectory.finish()
        return self.trajectory

    @staticmethod
    def parse_answer(message: Message) -> str:
        if message["role"] != "assistant":
            logger.error(f"Expected message role 'assistant', got {message['role']}")
            raise ValueError("Message role must be 'assistant' to extract answer.")

        content = message.get("content")
        if not isinstance(content, str):
            logger.error(f"Expected message content to be a string, got {type(content)}")
            raise ValueError("Message content must be a string.")

        return content
