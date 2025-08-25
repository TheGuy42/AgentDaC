from __future__ import annotations

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
from art import Trajectory

from src.utils import text as text_utils
from src.utils.visualize import trajectory_string, message_string
from src.utils.markers import Markers
from src.utils.logging import create_logger
from src.openai_types import Message, SystemMessage, UserMessage, AssistantMessage
from src.configs import PromptConfig, DecompConfig
from src.configs.prompts import get_prompt


logger = create_logger(__name__)


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
    return completion


class AgentNode:
    def __init__(
        self,
        openai_client: AsyncOpenAI,
        model_name: str,
        prompt_config: PromptConfig,
        decomp_config: DecompConfig,
        current_depth: int = 0,
    ):
        self.openai_client = openai_client
        self.model = model_name
        self.prompt_config = prompt_config
        self.decomp_config = decomp_config.clone()
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
            self.trajectory.messages_and_choices.append(sys_msg)

    @property
    def metrics(self) -> dict[str, float | int | bool]:
        return self.trajectory.metrics

    def __str__(self) -> str:
        return trajectory_string(self.trajectory)

    def _create_system_message(self) -> SystemMessage | None:
        prompt_config = self.prompt_config
        max_depth = self.decomp_config.max_depth

        if max_depth is None:
            max_depth = float("inf")

        content: str | None = None

        if self.decomp_config.should_stop(self.current_depth):
            # Leaf if we need to stop for any reason
            content = get_prompt(prompt_config.system_leaf)

        elif self.current_depth == 0:
            content = get_prompt(prompt_config.system_root)

        elif self.current_depth < max_depth:
            content = get_prompt(prompt_config.system_inter)

        if content is not None:
            return SystemMessage(role="system", content=content)

        return None

    def create_sub_agent(self) -> AgentNode:
        return AgentNode(
            openai_client=self.openai_client,
            model_name=self.model,
            prompt_config=self.prompt_config,
            decomp_config=self.decomp_config,
            current_depth=self.current_depth + 1,
        )

    async def _call(self, messages: list[Message], **kwargs) -> ChatCompletion:
        """
        Call the OpenAI API to get a chat completion.

        Args:
            messages (list[Message]): The list of messages to send to the API.
            **kwargs: Additional keyword arguments to pass to the API call.
        """
        # By default allow only a single task and answer in the response
        extra_body = kwargs.setdefault("extra_body", {})
        extra_body.setdefault("include_stop_str_in_output", True)
        kwargs.setdefault("stop", [Markers.TASK_END, Markers.ANSWER_END])

        completion = await self.openai_client.chat.completions.create(
            model=self.model,
            messages=messages,
            logprobs=True,
            **kwargs,
        )
        return patch_completion(completion)

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
        if prompt["role"] != "user":
            logger.warning(f"Prompt role is expected to be 'user', but got {prompt['role']}.")

        self.trajectory.messages_and_choices.append(prompt)

        if verbose:
            print(trajectory_string(self.trajectory, indent=self.current_depth))

        should_break = False

        while True:
            # Call the OpenAI API to get a response
            completion = await self._call(self.trajectory.messages(), **kwargs)
            self.trajectory.messages_and_choices.append(completion.choices[0])

            # Update metrics
            self.metrics["total_calls"] += 1
            self.metrics["direct_calls"] += 1
            if completion.usage:
                self.metrics["total_tokens"] = completion.usage.total_tokens

            if verbose:
                print(message_string(self.trajectory.messages()[-1], indent=self.current_depth))

            # Extract tasks from the response
            tasks_inputs = self.parse_tasks(self.trajectory.messages()[-1])

            if should_break or len(tasks_inputs) == 0:
                break  # No tasks to delegate, so last message

            task_responses: list[AssistantMessage] = []

            if self.decomp_config.should_stop(self.current_depth):
                mock_answer = get_prompt(self.prompt_config.tasks_depleted)
                if mock_answer is None:
                    break

                should_break = True
                for task in tasks_inputs:
                    # Provide mock answer indicating no more tasks available
                    task_responses.append(AssistantMessage(role="assistant", content=mock_answer))

            else:
                for task in tasks_inputs:
                    # create a sub-agent and get answer the task
                    sub_agent = self.create_sub_agent()
                    resp = await sub_agent.answer(task, verbose, **kwargs)
                    task_responses.append(resp)

                    # update metrics from sub-agent
                    self.metrics["total_tasks"] += sub_agent.metrics["total_tasks"]
                    self.metrics["total_calls"] += sub_agent.metrics["total_calls"]
                    self.metrics["max_depth"] = max(1 + sub_agent.metrics["max_depth"], self.metrics["max_depth"])

            # Update metrics
            self.metrics["direct_tasks"] += len(tasks_inputs)
            self.metrics["total_tasks"] += len(tasks_inputs)

            # Create a new message with all tasks' answers
            tasks_answers = [
                "{} {} {}".format(Markers.ANSWER_START, resp.get("content"), Markers.ANSWER_END)
                for resp in task_responses
            ]
            joined_message = UserMessage(role="user", content="\n".join(tasks_answers))
            self.trajectory.messages_and_choices.append(joined_message)

            if verbose:
                print(message_string(self.trajectory.messages()[-1], indent=self.current_depth))

            self.decomp_config.update_round(num_tasks=len(tasks_inputs))

        self.trajectory.finish()
        return self.trajectory

    async def answer(self, prompt: Message, verbose: bool = False, **kwargs) -> AssistantMessage:
        """
        Answer a question using the agent.

        Args:
            prompt (Message): The question to answer.
            verbose (bool): If True, print the conversation messages.
            **kwargs: Additional keyword arguments to pass to OpenAI API call.
        Returns:
            (AssistantMessage): The answer message from the agent.
        """
        trajectory = await self.chat(prompt, verbose=verbose, **kwargs)
        return self.parse_answer(trajectory.messages()[-1])

    def parse_answer(self, message: Message) -> AssistantMessage:
        if message["role"] != "assistant":
            logger.error(f"Expected message role 'assistant', got {message['role']}")
            raise ValueError("Message role must be 'assistant' to extract answer.")

        content = message.get("content")
        if not isinstance(content, str):
            logger.error(f"Expected message content to be a string, got {type(content).__name__}")
            raise ValueError("Message content must be a string.")

        answer = text_utils.extract_answer(content)
        return AssistantMessage(role="assistant", content=answer)

    def parse_tasks(self, message: Message) -> list[UserMessage]:
        if message["role"] != "assistant":
            raise ValueError("Message role must be 'assistant' to extract tasks.")

        content = message.get("content")
        if not isinstance(content, str):
            logger.error(f"Expected message content to be a string, got {type(content).__name__}")
            raise ValueError("Message content must be a string.")

        tasks = text_utils.extract_tasks(content)
        return [UserMessage(role="user", content=task) for task in tasks]
