from __future__ import annotations

from typing import cast
from openai import AsyncOpenAI
from openai.types.chat.chat_completion import ChatCompletion

from art import Trajectory
from art.types import Message
from pydantic import BaseModel, Field
from pathlib import Path

from src.utils import text as text_utils
from src.utils.visualize import trajectory_string, message_string
from src.configs.markers import Markers
from src.configs.prompts import get_prompt
from src.utils.logging import create_logger
from src.utils.io import save_base_model

logger = create_logger(__name__)


class ChatMessage(BaseModel, extra="allow", frozen=True, strict=True):
    role: str
    content: str

    def as_openai(self) -> Message:
        return cast(Message, {"role": self.role, "content": self.content})


class PromptConfig(BaseModel):
    system_root: str = ""
    system_inter: str = ""
    system_leaf: str = ""
    tasks_depleted: str = ""

    def save(self, dir_name: str, file_name: str = "prompt_config.json") -> None:
        """
        Save the prompt configuration to a JSON file.
        """
        save_base_model(self, Path(dir_name) / file_name)


class StopCriteria(BaseModel):
    max_depth: int | None = 1
    max_tasks: int | None = 5
    max_rounds: int | None = 5

    # Internal counter fields
    total_rounds: int = Field(default=0, exclude=True, init=False)
    total_tasks: int = Field(default=0, exclude=True, init=False)

    def clone(self) -> StopCriteria:
        """Create a deep copy and reset the counters"""
        new = self.model_copy(deep=True)
        new.reset()
        return new

    def reset(self):
        """Reset the internal counters"""
        self.total_rounds = 0
        self.total_tasks = 0

    def update_round(self, num_tasks: int):
        """Update round and task counters"""
        self.total_rounds += 1
        self.total_tasks += num_tasks

    def should_stop(self, cur_depth: int) -> bool:
        """Check if stopping criteria are met"""
        if self.max_depth and cur_depth >= self.max_depth:
            return True

        if self.max_tasks and self.total_tasks >= self.max_tasks:
            return True

        if self.max_rounds and self.total_rounds >= self.max_rounds:
            return True

        return False

    def save(self, dir_name: str, file_name: str = "stop_criteria.json") -> None:
        """
        Save the stop criteria configuration to a JSON file.
        """
        save_base_model(self, Path(dir_name) / file_name)


def patch_completion(completion: ChatCompletion) -> ChatCompletion:
    """
    Sometimes the OpenAI API returns choices with None content.
    It happens when the model response is immediately EOS.

    This function patches the completion to ensure all choices have content.
    If content is None, it sets it to an empty string.
    """
    for choice in completion.choices:
        if choice.message.content is None:
            choice.message.content = ""
            logger.warning("Patched choice with None 'message.content' to empty string.")
    return completion


class AgentNode:
    def __init__(
        self,
        openai_client: AsyncOpenAI,
        model_name: str,
        prompt_config: PromptConfig,
        stop_criteria: StopCriteria,
        current_depth: int = 0,
    ):
        self.openai_client = openai_client
        self.model = model_name
        self.prompt_config = prompt_config
        self.stop_criteria = stop_criteria.clone()
        self.current_depth = current_depth

        self.trajectory = Trajectory(
            messages_and_choices=[],
            reward=0,
            metrics={
                "direct_calls": 0,
                "total_calls": 0,
                "direct_tasks": 0,
                "total_tasks": 0,
                "max_depth": 0,
            },
        )

        if sys_msg := self._create_system_message():
            self.trajectory.messages_and_choices.append(sys_msg.as_openai())

    @property
    def metrics(self) -> dict[str, float | int | bool]:
        return self.trajectory.metrics

    def __str__(self) -> str:
        return trajectory_string(self.trajectory)

    def _create_system_message(self) -> ChatMessage | None:
        prompt_config = self.prompt_config
        max_depth = self.stop_criteria.max_depth

        if max_depth is None:
            max_depth = float("inf")

        content: str | None = None

        if self.stop_criteria.should_stop(self.current_depth):
            # Leaf if we need to stop for any reason
            content = get_prompt(prompt_config.system_leaf)

        elif self.current_depth == 0:
            content = get_prompt(prompt_config.system_root)

        elif self.current_depth < max_depth:
            content = get_prompt(prompt_config.system_inter)

        if content is not None:
            return ChatMessage(role="system", content=content)

        return None

    async def chat(
        self,
        prompt: ChatMessage,
        verbose: bool = False,
        **kwargs,
    ) -> Trajectory:
        """
        Start a conversation with the agent using the provided prompt.

        Args:
            prompt (ChatMessage): The initial message to start the conversation.
            verbose (bool): If True, print the conversation messages.
            **kwargs: Additional keyword arguments to pass to OpenAI API call.

        Returns:
            Trajectory: The trajectory of the conversation, including messages and choices.
                This trajectory is used to train an `art.TrainableModel` model.
        """

        if prompt.role != "user":
            raise ValueError("Prompt role must be 'user' to start the conversation.")

        self.trajectory.messages_and_choices.append(prompt.as_openai())

        if verbose:
            last_message = self.trajectory.messages()[-1]
            print(trajectory_string(self.trajectory, indent=self.current_depth))

        should_break = False

        while True:
            # Call the OpenAI API to get a response
            completion = await self._call(self.trajectory.messages(), **kwargs)
            choice = completion.choices[0]
            self.trajectory.messages_and_choices.append(choice)

            # Update metrics
            self.metrics["total_calls"] += 1
            self.metrics["direct_calls"] += 1
            if completion.usage:
                self.metrics["total_tokens"] = completion.usage.total_tokens

            if verbose:
                last_message = self.trajectory.messages()[-1]
                print(message_string(last_message, indent=self.current_depth))

            # Extract tasks from the response
            response = ChatMessage.model_validate(choice.message, from_attributes=True)
            tasks = self._parse_tasks(response)

            if should_break or len(tasks) == 0:
                break  # No tasks to delegate, so last message

            task_responses = []

            if self.stop_criteria.should_stop(self.current_depth):
                content = get_prompt(self.prompt_config.tasks_depleted)
                if content is None:
                    break

                # Provide mock answers indicating no more tasks available
                resp = ChatMessage(role="user", content=content)
                task_responses = [resp] * len(tasks)
                should_break = True

            else:
                for task in tasks:
                    # create a sub-agent and get answer the task
                    sub_agent = self.create_sub_agent()
                    resp = await sub_agent.answer(task, verbose, **kwargs)
                    task_responses.append(resp)

                    # update metrics from sub-agent
                    self.metrics["total_tasks"] += sub_agent.metrics["total_tasks"]
                    self.metrics["total_calls"] += sub_agent.metrics["total_calls"]
                    self.metrics["max_depth"] = max(1 + sub_agent.metrics["max_depth"], self.metrics["max_depth"])

            # Update metrics
            self.metrics["direct_tasks"] += len(tasks)
            self.metrics["total_tasks"] += len(tasks)

            # Format the task responses
            task_answers = []
            for resp in task_responses:
                answer_text = f"{Markers.ANSWER_START} {resp.content} {Markers.ANSWER_END}"
                task_answers.append(answer_text)

            # Create a new message with the tasks' answers
            tasks_message = ChatMessage(role="user", content="\n".join(task_answers))
            self.trajectory.messages_and_choices.append(tasks_message.as_openai())

            if verbose:
                last_message = self.trajectory.messages()[-1]
                print(message_string(last_message, indent=self.current_depth))

            self.stop_criteria.update_round(num_tasks=len(tasks))

        self.trajectory.finish()
        return self.trajectory

    async def answer(self, prompt: ChatMessage, verbose: bool = False, **kwargs) -> ChatMessage:
        """
        Answer a question using the agent.

        Args:
            prompt (ChatMessage): The question to answer.
            verbose (bool): If True, print the conversation messages.
            **kwargs: Additional keyword arguments to pass to OpenAI API call.
        Returns:
            ChatMessage: The answer message from the agent.
        """

        if prompt.role != "user":
            raise ValueError("Prompt role must be 'user' to start the conversation.")

        trajectory = await self.chat(prompt, verbose=verbose, **kwargs)
        last_message = ChatMessage.model_validate(trajectory.messages()[-1])
        return self._parse_answer(last_message)

    def create_sub_agent(self) -> AgentNode:
        return AgentNode(
            openai_client=self.openai_client,
            model_name=self.model,
            prompt_config=self.prompt_config,
            stop_criteria=self.stop_criteria,
            current_depth=self.current_depth + 1,
        )

    async def _call(self, messages: list[Message], **kwargs) -> ChatCompletion:
        """
        Call the OpenAI API to get a chat completion.

        Args:
            messages (list[Message]): The list of messages to send to the API.
            **kwargs: Additional keyword arguments to pass to the API call.
        """
        completion = await self.openai_client.chat.completions.create(
            model=self.model,
            messages=messages,
            logprobs=True,
            **kwargs,
        )

        return patch_completion(completion)

    def _parse_answer(self, message: ChatMessage) -> ChatMessage:
        if message.role != "assistant":
            raise ValueError("Message role must be 'assistant' to extract answer.")

        answer = text_utils.extract_answer(message.content)
        return ChatMessage(role="assistant", content=answer)

    def _parse_tasks(self, message: ChatMessage) -> list[ChatMessage]:
        if message.role != "assistant":
            raise ValueError("Message role must be 'assistant' to extract tasks.")

        tasks = text_utils.extract_tasks(message.content)
        tasks_messages = [ChatMessage(role="user", content=task) for task in tasks]
        return tasks_messages
