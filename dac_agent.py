from openai import AsyncOpenAI
from typing import Any, Dict, Optional, NewType
from openai.types.chat.chat_completion import ChatCompletion, Choice
from art import Trajectory

# define ChatMessage type
ChatMessage = NewType("ChatMessage", Dict[str, str])


class DACAgent:
    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        system_message: Optional[str] = None,
    ):
        self.client = client
        self.model = model
        self.sub_agents: list[DACAgent] = []
        self.trajectory: Trajectory = Trajectory()

    def chat(self, message: ChatMessage) -> Trajectory:
        self.trajectory.messages_and_choices.append(message)

        while True:
            messages = self.trajectory.messages()
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                # max_completion_tokens=128,
                logprobs=True,
            )

            tasks = self.parse_response(response)
            if len(tasks) == 0:
                break

            for task in tasks:
                # TODO: Implement the logic to handle aggregation of multiple sub-tasks
                sub_agent_response = self.call_sub_agent(task)
                self.trajectory.messages_and_choices.append(sub_agent_response)

        return self.trajectory

    def call_sub_agent(self, message: ChatMessage) -> ChatMessage:
        sub_agent: DACAgent = DACAgent(self.client, self.model)
        trajectory = sub_agent.chat(message)
        response = trajectory.messages()[-1]
        response['role'] = 'user'
        return response

    def parse_response(self, response: ChatMessage) -> list[ChatMessage]:
        """
        Parse the response from the chat completion and return a list of ChatMessage.
        Each ChatMessage should contain the role and content of the message to be passed to a sub-agent for further processing.
        One ChatMessage for each sub-task/sub-agent.

        :param response: The ChatMessage response from the sub-agent.
        :return: A list of ChatMessage containing the role and content for each sub-agent.
        """
        # content = 
        pass

    