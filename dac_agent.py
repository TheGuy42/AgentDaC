from openai import AsyncOpenAI, BadRequestError
from typing import Any, Dict, Optional, NewType, TypedDict
from openai.types.chat.chat_completion import ChatCompletion, Choice
from art import Trajectory
import re
from debug_utils import print_trajectory
import math

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
        leaf_sys_prompt: str = "",
        max_depth: int = 10,
        max_length: int = 20,
    ):
        self.client = client
        self.model = model
        self.model_system_message = model_system_message
        self.dac_sys_prompt = dac_sys_prompt
        self.leaf_sys_prompt = leaf_sys_prompt
        self.max_depth = max_depth
        self.max_length = max_length

        sys_prompt = model_system_message + "\n" + (dac_sys_prompt if self.max_depth > 0 else leaf_sys_prompt)
        self.system_message = {
            "role": "system",
            # "role": "user",
            "content": sys_prompt,
        }
        self.sub_agents: list[DACAgent] = []
        self.trajectory: Trajectory = Trajectory(
            messages_and_choices=[self.system_message], reward=0
        )

        self.id = DACAgent.counter
        DACAgent.counter += 1  # Increment the counter for each new instance

    async def chat(self, message: ChatMessage, **kwargs) -> Trajectory:
        counter = 0

        self.trajectory.messages_and_choices.append(message)
        while True:
            counter += 1

            messages = self.trajectory.messages()
            if counter > self.max_length:
                messages[-1]["content"] += "\n[Please provide your final answer to the original question.]"
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    logprobs=True,
                    # stop="</task>",  # Stop at the end of a task
                    **kwargs,  # Additional keyword arguments for the chat completion
                    # max_completion_tokens=9900,  # Set a high limit to allow for long responses
                )
                self.trajectory.messages_and_choices.append(response.choices[0])
                self.trajectory.reward += self.format_reward(response.choices[0].message.content)
            except BadRequestError as e:
                print(f"BadRequestError: {e}")
                break
            # If max_depth is 0, or if we reached the max length, we do not delegate to sub-agents
            if self.max_depth <= 0 or counter > self.max_length:
                break
            
            # Parse the response to extract sub-tasks
            tasks = self.parse_response(response)
            if len(tasks) == 0:
                break
            
            # delegate tasks to sub-agents
            task_responses = []
            for task in tasks:
                # TODO: Implement the logic to handle aggregation of multiple sub-tasks
                sub_agent_response = await self.call_sub_agent(task)
                task_responses.append(sub_agent_response)
            
            # Add all sub-agent responses to the trajectory
            if len(task_responses) > 0:
                # concatenate all sub-agent responses into a single message
                combined_response = {
                    "role": "user",
                    "content": "".join(
                        [f"{resp['content']}" for resp in task_responses]
                    ),
                }
                self.trajectory.messages_and_choices.append(combined_response)

        return self.trajectory

    async def call_sub_agent(self, message: ChatMessage) -> ChatMessage:
        sub_agent: DACAgent = DACAgent(
            self.client,
            self.model,
            self.model_system_message,
            dac_sys_prompt=self.dac_sys_prompt,
            leaf_sys_prompt=self.leaf_sys_prompt,
            max_depth=self.max_depth - 1,
            max_length=self.max_length,
        )
        trajectory = await sub_agent.chat(message)
        response = trajectory.messages()[-1]
        # Extract the answer from the sub-agent response
        answer = extract_text_between_markers(
            response["content"], "<answer>", "</answer>"
        )
        if len(answer) == 0:
            # if self.max_depth > 0:
            #     self.trajectory.reward -= 0.1  # Penalize for no answer formatting
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
    
    def format_reward(self, response: str) -> float:
        reward = 0.0

        tasks = extract_text_between_markers(response, "<task>", "</task>")
        answers = extract_text_between_markers(response, "<answer>", "</answer>")
        
        if len(tasks) == 0 and len(answers) == 0:
            reward -= 0.1  # Penalize for no tasks or answers
        elif len(tasks) > 0 and len(answers) == 0:
            reward += 0.2 #** len(tasks)  # Reward for tasks without answers
        elif len(tasks) == 0 and len(answers) > 0:
            reward += 0.2 ** len(answers)  # Reward for answers without tasks, diminishing with more answers
        else:
            reward -= 0.1 ** (1 / min(len(tasks), len(answers)))  # Penalize for each task that was also answered by the agent
        
        tasks_diff = abs(response.count("<task>") - response.count("</task>"))
        answers_diff = abs(response.count("<answer>") - response.count("</answer>"))
        if tasks_diff > 0:
            reward -= 0.1 ** (1 / tasks_diff)
        if answers_diff > 0:
            reward -= 0.1 ** (1 / answers_diff)

        return reward


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
