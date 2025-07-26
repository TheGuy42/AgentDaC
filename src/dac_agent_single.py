from openai.types.chat.chat_completion import ChatCompletion
from art.types import Message

from src.dac_agent import AgentNode
from src.configs.markers import Markers

class SingleAgentNode(AgentNode):
    def create_sub_agent(self):
        return SingleAgentNode(
            client=self.client,
            model_name=self.model,
            prompt_config=self.prompt_config,
            stop_criteria=self.stop_criteria.clone(),
            cur_depth=self.cur_depth + 1,
        )

    async def _call(self, messages: list[Message], **kwargs) -> ChatCompletion:
        return await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            logprobs=True,
            stop=[Markers.TASK_END, Markers.ANSWER_END],
            extra_body={"include_stop_str_in_output": True},
            **kwargs,
        )
