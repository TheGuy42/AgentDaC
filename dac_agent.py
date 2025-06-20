from openai import AsyncOpenAI
from typing import Any, Dict, Optional, NewType, TypedDict
from openai.types.chat.chat_completion import ChatCompletion, Choice
from art import Trajectory
import re
from debug_utils import print_trajectory

# define ChatMessage type
# ChatMessage = NewType("ChatMessage", Dict[str, str])
ChatMessage = TypedDict(
    "ChatMessage",
    {
        "role": str,  # "user", "assistant", or "system"
        "content": str,  # The content of the message
        # "name": Optional[str],  # Optional name for the message sender
    },
    # total=False,
)


class DACAgent:
    counter = 0  # Static counter to keep track of agent instances
    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        model_system_message: Optional[str] = "",
        dac_sys_prompt: str = "",
        max_depth: int = 10,
    ):
        self.client = client
        self.model = model
        self.model_system_message = model_system_message
        self.dac_sys_prompt = dac_sys_prompt
        self.max_depth = max_depth

        self.system_message = {
            "role": "system",
            # "role": "user",
            "content": model_system_message + "\n" + dac_sys_prompt,
        }
        self.sub_agents: list[DACAgent] = []
        self.trajectory: Trajectory = Trajectory(
            messages_and_choices=[self.system_message], reward=0
        )

        self.id = DACAgent.counter
        DACAgent.counter += 1  # Increment the counter for each new instance

        # initialize the trajectory with the system message
        # self.trajectory.messages_and_choices.append(self.system_message)

    async def chat(self, message: ChatMessage) -> Trajectory:
        self.trajectory.messages_and_choices.append(message)
        # pad = " " * (self.max_depth * 2)
        # print(f"Chat from agent {self.id} at depth {self.max_depth}:")
        while True:
            messages = self.trajectory.messages()
            # print(f"{pad}- {messages[-1]['role']}:: {messages[-1]['content']}")
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                # max_completion_tokens=128,
                logprobs=True,
            )
            self.trajectory.messages_and_choices.append(response.choices[0])
            # print(f"{pad}- {self.trajectory.messages()[-1]['content']}")

            if self.max_depth <= 0:
                break
            # If max_depth is 0, we do not delegate to sub-agents
            tasks = self.parse_response(response)
            if len(tasks) == 0:
                break

            for task in tasks:
                # TODO: Implement the logic to handle aggregation of multiple sub-tasks
                sub_agent_response = await self.call_sub_agent(task)
                self.trajectory.messages_and_choices.append(sub_agent_response)

        return self.trajectory

    async def call_sub_agent(self, message: ChatMessage) -> ChatMessage:
        sub_agent: DACAgent = DACAgent(
            self.client,
            self.model,
            self.model_system_message,
            dac_sys_prompt=self.dac_sys_prompt if (self.max_depth - 1) > 0 else "",
            max_depth=self.max_depth - 1,
        )
        trajectory = await sub_agent.chat(message)
        # print(f"Sub-agent trajectory for depth {self.max_depth - 1}:")
        # print_trajectory(trajectory)
        response = trajectory.messages()[-1]
        # Extract the answer from the sub-agent response
        answer = extract_text_between_markers(
            response["content"], "<answer>", "</answer>"
        )
        if len(answer) == 0:
            if self.max_depth > 0:
                self.trajectory.reward -= 0.1  # Penalize for no answer formatting
            answer = response["content"]
        else:
            # combine all answers if there are multiple
            answer = " ".join(answer)
        response["role"] = "user"
        response["content"] = f"<answer> {answer} </answer>"
        return response

    def parse_response(self, response: ChatCompletion) -> list[ChatMessage]:
        """
        Parse the response from the chat completion and return a list of ChatMessage.
        Each ChatMessage should contain the role and content of the message to be passed to a sub-agent for further processing.
        One ChatMessage for each sub-task/sub-agent.

        :param response: The ChatCompletion response from the sub-agent.
        :return: A list of ChatMessage containing the role and content for each sub-agent.
        """
        content = response.choices[0].message.content
        if not content:
            return []
        # Extract all <task>...</task> blocks from the content
        tasks = extract_text_between_markers(content, "<task>", "</task>")
        tasks_messages = []
        for task in tasks:
            tasks_messages.append(ChatMessage({"role": "user", "content": task}))

        return tasks_messages


def extract_text_between_markers(
    text: str, start_marker: str, end_marker: str
) -> list[str]:
    """
    Extracts all instances of text between two specific markers in a string.

    Args:
        text: The input string to parse.
        start_marker: The beginning marker.
        end_marker: The ending marker.

    Returns:
        A list of strings, where each string is an instance of text found between the markers.
    """
    # Create a regex pattern to find text non-greedily between the markers
    # re.escape is used to escape any special characters in the markers
    # (.*?) matches any character (except newline) zero or more times, non-greedily
    pattern = re.escape(start_marker) + r"(.*?)" + re.escape(end_marker)

    # Find all non-overlapping matches of the pattern in the string
    matches = re.findall(pattern, text, re.DOTALL)

    return matches
