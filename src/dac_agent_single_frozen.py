from openai.types.chat.chat_completion import ChatCompletion
from art.types import Message
from openai import AsyncOpenAI

from src.dac_agent import AgentNode, patch_completion, ChatMessage, PromptConfig, StopCriteria
from src.configs.markers import Markers
from src.dac_agent import AgentNode, patch_completion
from src.configs.markers import Markers


class SingleAgentNodeFrozen(AgentNode):
    def __init__(
        self,
        openai_client: AsyncOpenAI,
        model_name: str,
        prompt_config: PromptConfig,
        stop_criteria: StopCriteria,
        current_depth: int = 0,
        include_histories: bool = False,
        base_model_name: str | None = None,
    ):
        super().__init__(
            openai_client=openai_client,
            model_name=model_name,
            prompt_config=prompt_config,
            stop_criteria=stop_criteria,
            current_depth=current_depth,
            include_histories=include_histories,
        )
        self.base_model_name = base_model_name

    def create_sub_agent(self):
        model_name = self.base_model_name if self.base_model_name else self.model
        return SingleAgentNodeFrozen(
            openai_client=self.openai_client,
            model_name=model_name,
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

        return completion
        return patch_completion(completion)
