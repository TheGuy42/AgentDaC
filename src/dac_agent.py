from openai import AsyncOpenAI
from openai.types.chat.chat_completion import ChatCompletion

from art import Trajectory
from art.types import Message
from dataclasses import dataclass

import copy
from typing import TypedDict, Required
from pydantic import BaseModel

from src.utils.text_utils import extract_text_between_markers
from src.utils.visualize import trajectory_string, message_string
from src.configs.markers import Markers


@dataclass
class SysPrompt:
    root_prompt: str | None = None
    inter_prompt: str | None = None
    leaf_prompt: str | None = None


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


class ChatMessage(BaseModel, extra="allow", frozen=True, strict=True):
    role: str
    content: str


class AgentNode:
    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        system_prompt: SysPrompt,
        stop_criteria: StopCriteria,
        cur_depth: int = 0,
    ):
        self.client = client
        self.model = model
        self.system_prompt = system_prompt
        self.stop_criteria = stop_criteria

        self.cur_depth = cur_depth

        self.metrics = Metrics(
            direct_calls=0,
            total_calls=0,
            direct_tasks=0,
            total_tasks=0,
        )

        self.trajectory = Trajectory(messages_and_choices=[], reward=0)

        sys_msg = self._create_system_message(system_prompt)
        if sys_msg is not None:
            self.trajectory.messages_and_choices.append(sys_msg.model_dump())

    def __str__(self) -> str:
        return trajectory_string(self.trajectory)

    def _create_system_message(self, system_prompt: SysPrompt) -> ChatMessage | None:
        max_depth = self.stop_criteria.max_depth
        if max_depth is None:
            max_depth = float("inf")

        if self.cur_depth == 0 and system_prompt.root_prompt:
            return ChatMessage(role="system", content=system_prompt.root_prompt)
        elif self.cur_depth < max_depth and system_prompt.inter_prompt:
            return ChatMessage(role="system", content=system_prompt.inter_prompt)
        elif self.cur_depth >= max_depth and system_prompt.leaf_prompt:
            return ChatMessage(role="system", content=system_prompt.leaf_prompt)
        return None

    async def chat(
        self,
        prompt: ChatMessage,
        verbose: bool = False,
        **kwargs,
    ) -> Trajectory:
        self.trajectory.messages_and_choices.append(prompt.model_dump())
        
        if verbose:
            print(trajectory_string(self.trajectory, indent=self.cur_depth))

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

            if self.stop_criteria.should_stop(self.cur_depth):
                break  # stop If we reached sub-agent delegation limit

            # Extract tasks from the response
            response = ChatMessage.model_validate(choice.message, from_attributes=True)
            tasks = self._parse_tasks(response)
            self.stop_criteria.update_round(num_tasks=len(tasks))

            if len(tasks) == 0:
                break  # No tasks to delegate, so last message

            tasks_answers = []
            for task in tasks:
                # create a sub-agent and get answer the task
                sub_agent = self._create_agent()
                task_resp = await sub_agent.answer(task, verbose, **kwargs)
                answer_text = f"{Markers.ANSWER_START}{task_resp.content}{Markers.ANSWER_END}"
                tasks_answers.append(answer_text)

                # update metrics from sub-agent
                self.metrics["direct_tasks"] += 1
                self.metrics["total_tasks"] += 1 + sub_agent.metrics["total_tasks"]
                self.metrics["total_calls"] += sub_agent.metrics["total_calls"]

            # Create a new message with the tasks' answers
            tasks_message = ChatMessage(role="user", content="\n".join(tasks_answers))
            self.trajectory.messages_and_choices.append(tasks_message.model_dump())

            if verbose:
                last_message = self.trajectory.messages()[-1]
                print(message_string(last_message, indent=self.cur_depth))

        self.trajectory.metrics.update(self.metrics)
        return self.trajectory

    async def answer(self, prompt: ChatMessage, verbose: bool = False, **kwargs) -> ChatMessage:
        if prompt.role != "user":
            raise ValueError("Prompt role must be 'user' to start the conversation.")

        trajectory = await self.chat(prompt, verbose=verbose, **kwargs)
        last_message = ChatMessage.model_validate(trajectory.messages()[-1])
        return self._parse_answer(last_message)

    def _create_agent(self):
        stop = copy.deepcopy(self.stop_criteria)
        stop.reset()  # Reset the stop criteria for the new agent

        return AgentNode(
            client=self.client,
            model=self.model,
            system_prompt=self.system_prompt,
            stop_criteria=stop,
            cur_depth=self.cur_depth + 1,
        )

    async def _call(self, messages: list[Message], **kwargs) -> ChatCompletion:
        return await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            logprobs=True,
            **kwargs,
        )

    def _parse_answer(self, message: ChatMessage) -> ChatMessage:
        if message.role != "assistant":
            raise ValueError("Message role must be 'assistant' to extract answer.")

        content = message.content
        answer_list = extract_text_between_markers(content, Markers.ANSWER_START, Markers.ANSWER_END)
        if len(answer_list) > 0:
            content = answer_list[-1]  # Take the last answer if multiple are found
        content = content.strip()
        return ChatMessage(role="assistant", content=content)

    def _parse_tasks(self, message: ChatMessage) -> list[ChatMessage]:
        if message.role != "assistant":
            raise ValueError("Message role must be 'assistant' to extract tasks.")

        content = message.content
        tasks = extract_text_between_markers(content, Markers.TASK_START, Markers.TASK_END)
        tasks_messages = []
        for task in tasks:
            tasks_messages.append(ChatMessage(role="user", content=task.strip()))

        return tasks_messages
