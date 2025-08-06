from openai.types.chat.chat_completion import ChatCompletion
from art.types import Message

from src.dac_agent import AgentNode, patch_completion
from src.configs.markers import Markers


class SingleAgentNode(AgentNode):
    def create_sub_agent(self):
        return SingleAgentNode(
            openai_client=self.openai_client,
            model_name=self.model,
            prompt_config=self.prompt_config,
            stop_criteria=self.stop_criteria,
            current_depth=self.current_depth + 1,
        )

    async def _call(self, messages: list[Message], **kwargs) -> ChatCompletion:
        # format kwargs
        extra_body = kwargs.pop("extra_body", {})
        extra_body.setdefault("include_stop_str_in_output", True)
        stop = kwargs.pop("stop", [Markers.TASK_END, Markers.ANSWER_END])

        completion = await self.openai_client.chat.completions.create(
            model=self.model,
            messages=messages,
            logprobs=True,
            stop=stop,
            extra_body=extra_body,
            **kwargs,
        )

        return patch_completion(completion)
