from __future__ import annotations

from openai import AsyncOpenAI
from openai.types.chat.chat_completion import ChatCompletion

from art import Trajectory
from art.types import Message

import copy
from typing import TypedDict, Required
from pydantic import BaseModel

from src.utils import text as text_utils
from src.utils.visualize import trajectory_string, message_string
from src.configs.markers import Markers

# TODO: think about the structure and the flow of the chat method, especially
# of the chat() method. we should warn in advance if the task budget is about
# to deplete, i.e instruct the model to provide a final answer.


class ChatMessage(BaseModel, extra="allow", frozen=True, strict=True):
    role: str
    content: str


class PromptConfig:
    def __init__(
        self,
        root_system_prompt: str | None = None,
        inter_system_prompt: str | None = None,
        leaf_system_prompt: str | None = None,
        tasks_depleted_prompt: str = "No more tasks available, please answer the question directly.",
    ):
        self.root_prompt: ChatMessage | None = None
        if root_system_prompt is not None:
            self.root_prompt = ChatMessage(role="system", content=root_system_prompt)

        self.inter_prompt: ChatMessage | None = None
        if inter_system_prompt is not None:
            self.inter_prompt = ChatMessage(role="system", content=inter_system_prompt)

        self.leaf_prompt: ChatMessage | None = None
        if leaf_system_prompt is not None:
            self.leaf_prompt = ChatMessage(role="system", content=leaf_system_prompt)

        self.tasks_depleted_prompt = ChatMessage(role="user", content=tasks_depleted_prompt)

    # TODO: we should add a message which is returned in case of forking budget running out.


class StopCriteria:
    def __init__(
        self,
        max_depth: int | None = 1,
        max_tasks: int | None = 5,
        max_rounds: int | None = 5,
    ):
        self.max_depth = max_depth
        self.max_tasks = max_tasks
        self.max_rounds = max_rounds

        self._total_rounds = 0
        self._total_tasks = 0

    def clone(self):
        new = copy.deepcopy(self)
        new.reset()  # Reset the counters for the cloned instance
        return new

    def reset(self):
        self._total_rounds = 0
        self._total_tasks = 0

    def update_round(self, num_tasks: int):
        self._total_rounds += 1
        self._total_tasks += num_tasks

    def should_stop(self, cur_depth: int) -> bool:
        if self.max_depth and cur_depth >= self.max_depth:
            return True

        if self.max_tasks and self._total_tasks >= self.max_tasks:
            return True

        if self.max_rounds and self._total_rounds >= self.max_rounds:
            return True

        return False


class Metrics(TypedDict, total=False):
    direct_calls: Required[int]
    total_calls: Required[int]
    direct_tasks: Required[int]
    total_tasks: Required[int]


class AgentNode:
    def __init__(
        self,
        client: AsyncOpenAI,
        model_name: str,
        prompt_config: PromptConfig,
        stop_criteria: StopCriteria,
        cur_depth: int = 0,
    ):
        self.client = client
        self.model = model_name
        self.prompt_config = prompt_config
        self.stop_criteria = stop_criteria.clone()

        self.cur_depth = cur_depth

        self.metrics = Metrics(
            direct_calls=0,
            total_calls=0,
            direct_tasks=0,
            total_tasks=0,
        )

        self.trajectory = Trajectory(messages_and_choices=[], reward=0)

        sys_msg = self._create_system_message(prompt_config)
        if sys_msg is not None:
            self.trajectory.messages_and_choices.append(sys_msg.model_dump())

    def __str__(self) -> str:
        return trajectory_string(self.trajectory)

    def _create_system_message(self, system_prompt: PromptConfig) -> ChatMessage | None:
        max_depth = self.stop_criteria.max_depth

        if max_depth is None:
            max_depth = float("inf")

        if self.stop_criteria.should_stop(self.cur_depth):
            # We're a leaf if we need to stop for any reason
            return system_prompt.leaf_prompt
        if self.cur_depth == 0:
            return system_prompt.root_prompt
        if self.cur_depth < max_depth:
            return system_prompt.inter_prompt
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

        self.trajectory.messages_and_choices.append(prompt.model_dump())

        if verbose:
            last_message = self.trajectory.messages()[-1]
            print(message_string(last_message, indent=self.cur_depth))

        while True:
            # Call the OpenAI API to get a response
            completion = await self._call(self.trajectory.messages(), **kwargs)
            choice = completion.choices[0]
            self.trajectory.messages_and_choices.append(choice)

            # Update metrics
            self.metrics["total_calls"] += 1
            self.metrics["direct_calls"] += 1

            if verbose:
                last_message = self.trajectory.messages()[-1]
                print(message_string(last_message, indent=self.cur_depth))

            # Extract tasks from the response
            response = ChatMessage.model_validate(choice.message, from_attributes=True)
            tasks = self._parse_tasks(response)

            if len(tasks) == 0:
                break  # No tasks to delegate, so last message

            task_responses = []

            if self.stop_criteria.should_stop(self.cur_depth):
                # Provide mock answers indicating no more tasks available
                resp = self.prompt_config.tasks_depleted_prompt
                task_responses = [resp] * len(tasks)

            else:
                for task in tasks:
                    # create a sub-agent and get answer the task
                    sub_agent = self.create_sub_agent()
                    resp = await sub_agent.answer(task, verbose, **kwargs)
                    task_responses.append(resp)

                    # update metrics from sub-agent
                    self.metrics["direct_tasks"] += 1
                    self.metrics["total_tasks"] += 1 + sub_agent.metrics["total_tasks"]
                    self.metrics["total_calls"] += sub_agent.metrics["total_calls"]

            # Format the task responses
            task_answers = []
            for resp in task_responses:
                answer_text = f"{Markers.ANSWER_START} {resp.content} {Markers.ANSWER_END}"
                task_answers.append(answer_text)

            # Create a new message with the tasks' answers
            tasks_message = ChatMessage(role="user", content="\n".join(task_answers))
            self.trajectory.messages_and_choices.append(tasks_message.model_dump())

            if verbose:
                last_message = self.trajectory.messages()[-1]
                print(message_string(last_message, indent=self.cur_depth))

            self.stop_criteria.update_round(num_tasks=len(tasks))

        self.trajectory.metrics.update(self.metrics)
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

    def create_sub_agent(self):
        return AgentNode(
            client=self.client,
            model_name=self.model,
            prompt_config=self.prompt_config,
            stop_criteria=self.stop_criteria.clone(),
            cur_depth=self.cur_depth + 1,
        )

    async def _call(self, messages: list[Message], **kwargs) -> ChatCompletion:
        """
        Call the OpenAI API to get a chat completion.

        Args:
            messages (list[Message]): The list of messages to send to the API.
            **kwargs: Additional keyword arguments to pass to the API call.
        """
        return await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            logprobs=True,
            **kwargs,
        )

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
