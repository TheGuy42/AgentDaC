from __future__ import annotations
from typing import Any
import copy
import openai

from src.aliases import Message, Response, Choice
from src.inference.client import InferenceClient, InferenceResponse


class OAIResponse(InferenceResponse):
    def __init__(self, openai_response: Response) -> None:
        self.openai_response = openai_response

    @property
    def choice(self) -> Choice:
        return self.openai_response.choices[0]

    @property
    def content(self) -> str | None:
        return self.choice.message.content

    @property
    def tool_calls(self) -> list[Any] | None:
        return self.choice.message.tool_calls

    @property
    def reasoning(self) -> str | None:
        result = getattr(self.choice.message, "reasoning", None)
        if result is None:
            # NOTE: compatibility with older vLLM versions
            result = getattr(self.choice.message, "reasoning_content", None)
        return result

    @property
    def finish_reason(self) -> str | None:
        return self.choice.finish_reason

    @property
    def total_tokens(self) -> int | None:
        usage = self.openai_response.usage
        return usage.total_tokens if usage is not None else None


class OAIClient(InferenceClient):
    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = "EMPTY",
        client: openai.AsyncOpenAI | None = None,
        **kwargs,
    ):
        self.client = client or openai.AsyncOpenAI(base_url=base_url, api_key=api_key, **kwargs)
        self.model_name = model_name
        self._owns_client = client is None

    async def list_models(self) -> list[openai.types.Model]:
        page = await self.client.models.list()
        return page.data

    def update_kwargs(
        self,
        kwargs: dict[str, Any],
        json_schema: dict | None = None,
        regex_schema: str | None = None,
        include_stop_str_in_output: bool | None = None,
    ) -> dict[str, Any]:

        if json_schema is not None and regex_schema is not None:
            raise ValueError("Cannot specify both json_schema and regex_schema.")

        kwargs = copy.deepcopy(kwargs)

        extra_body: dict = kwargs.setdefault("extra_body", {})

        if json_schema is not None:
            kwargs["response_format"] = {"type": "json_schema", "json_schema": json_schema}

        if regex_schema is not None:
            extra_body["structured_outputs"] = {"regex": regex_schema}

        if include_stop_str_in_output is not None:
            extra_body["include_stop_str_in_output"] = include_stop_str_in_output

        return kwargs

    async def chat(self, messages: list[Message], **kwargs) -> InferenceResponse:
        """
        Creates a model response for the given chat conversation. Learn more in the
        [text generation](https://platform.openai.com/docs/guides/text-generation),
        [vision](https://platform.openai.com/docs/guides/vision), and
        [audio](https://platform.openai.com/docs/guides/audio) guides.

        Parameter support can differ depending on the model used to generate the
        response, particularly for newer reasoning models. Parameters that are only
        supported for reasoning models are noted below. For the current state of
        unsupported parameters in reasoning models,
        [refer to the reasoning guide](https://platform.openai.com/docs/guides/reasoning).

        Returns a chat completion object.

        Args:
          messages: A list of messages comprising the conversation so far. Depending on the
              [model](https://platform.openai.com/docs/models) you use, different message
              types (modalities) are supported, like
              [text](https://platform.openai.com/docs/guides/text-generation),
              [images](https://platform.openai.com/docs/guides/vision), and
              [audio](https://platform.openai.com/docs/guides/audio).

          audio: Parameters for audio output. Required when audio output is requested with
              `modalities: ["audio"]`.
              [Learn more](https://platform.openai.com/docs/guides/audio).

          frequency_penalty: Number between -2.0 and 2.0. Positive values penalize new tokens based on their
              existing frequency in the text so far, decreasing the model's likelihood to
              repeat the same line verbatim.

          function_call: Deprecated in favor of `tool_choice`.

              Controls which (if any) function is called by the model.

              `none` means the model will not call a function and instead generates a message.

              `auto` means the model can pick between generating a message or calling a
              function.

              Specifying a particular function via `{"name": "my_function"}` forces the model
              to call that function.

              `none` is the default when no functions are present. `auto` is the default if
              functions are present.

          functions: Deprecated in favor of `tools`.

              A list of functions the model may generate JSON inputs for.

          logit_bias: Modify the likelihood of specified tokens appearing in the completion.

              Accepts a JSON object that maps tokens (specified by their token ID in the
              tokenizer) to an associated bias value from -100 to 100. Mathematically, the
              bias is added to the logits generated by the model prior to sampling. The exact
              effect will vary per model, but values between -1 and 1 should decrease or
              increase likelihood of selection; values like -100 or 100 should result in a ban
              or exclusive selection of the relevant token.

          logprobs: Whether to return log probabilities of the output tokens or not. If true,
              returns the log probabilities of each output token returned in the `content` of
              `message`.

          max_completion_tokens: An upper bound for the number of tokens that can be generated for a completion,
              including visible output tokens and
              [reasoning tokens](https://platform.openai.com/docs/guides/reasoning).

          max_tokens: The maximum number of [tokens](/tokenizer) that can be generated in the chat
              completion. This value can be used to control
              [costs](https://openai.com/api/pricing/) for text generated via API.

              This value is now deprecated in favor of `max_completion_tokens`, and is not
              compatible with
              [o-series models](https://platform.openai.com/docs/guides/reasoning).

          metadata: Set of 16 key-value pairs that can be attached to an object. This can be useful
              for storing additional information about the object in a structured format, and
              querying for objects via API or the dashboard.

              Keys are strings with a maximum length of 64 characters. Values are strings with
              a maximum length of 512 characters.

          modalities: Output types that you would like the model to generate. Most models are capable
              of generating text, which is the default:

              `["text"]`

              The `gpt-4o-audio-preview` model can also be used to
              [generate audio](https://platform.openai.com/docs/guides/audio). To request that
              this model generate both text and audio responses, you can use:

              `["text", "audio"]`

          moderation: Configuration for running moderation on the request input and generated output.

          n: How many chat completion choices to generate for each input message. Note that
              you will be charged based on the number of generated tokens across all of the
              choices. Keep `n` as `1` to minimize costs.

          parallel_tool_calls: Whether to enable
              [parallel function calling](https://platform.openai.com/docs/guides/function-calling#configuring-parallel-function-calling)
              during tool use.

          prediction: Static predicted output content, such as the content of a text file that is
              being regenerated.

          presence_penalty: Number between -2.0 and 2.0. Positive values penalize new tokens based on
              whether they appear in the text so far, increasing the model's likelihood to
              talk about new topics.

          prompt_cache_key: Used by OpenAI to cache responses for similar requests to optimize your cache
              hit rates. Replaces the `user` field.
              [Learn more](https://platform.openai.com/docs/guides/prompt-caching).

          prompt_cache_retention: The retention policy for the prompt cache. Set to `24h` to enable extended
              prompt caching, which keeps cached prefixes active for longer, up to a maximum
              of 24 hours.
              [Learn more](https://platform.openai.com/docs/guides/prompt-caching#prompt-cache-retention).
              For `gpt-5.5`, `gpt-5.5-pro`, and future models, only `24h` is supported.

              For older models that support both `in_memory` and `24h`, the default depends on
              your organization's data retention policy:

              - Organizations without ZDR enabled default to `24h`.
              - Organizations with ZDR enabled default to `in_memory` when
                `prompt_cache_retention` is not specified.

          reasoning_effort: Constrains effort on reasoning for
              [reasoning models](https://platform.openai.com/docs/guides/reasoning). Currently
              supported values are `none`, `minimal`, `low`, `medium`, `high`, and `xhigh`.
              Reducing reasoning effort can result in faster responses and fewer tokens used
              on reasoning in a response.

              - `gpt-5.1` defaults to `none`, which does not perform reasoning. The supported
                reasoning values for `gpt-5.1` are `none`, `low`, `medium`, and `high`. Tool
                calls are supported for all reasoning values in gpt-5.1.
              - All models before `gpt-5.1` default to `medium` reasoning effort, and do not
                support `none`.
              - The `gpt-5-pro` model defaults to (and only supports) `high` reasoning effort.
              - `xhigh` is supported for all models after `gpt-5.1-codex-max`.

          response_format: An object specifying the format that the model must output.

              Setting to `{ "type": "json_schema", "json_schema": {...} }` enables Structured
              Outputs which ensures the model will match your supplied JSON schema. Learn more
              in the
              [Structured Outputs guide](https://platform.openai.com/docs/guides/structured-outputs).

              Setting to `{ "type": "json_object" }` enables the older JSON mode, which
              ensures the message the model generates is valid JSON. Using `json_schema` is
              preferred for models that support it.

          safety_identifier: A stable identifier used to help detect users of your application that may be
              violating OpenAI's usage policies. The IDs should be a string that uniquely
              identifies each user, with a maximum length of 64 characters. We recommend
              hashing their username or email address, in order to avoid sending us any
              identifying information.
              [Learn more](https://platform.openai.com/docs/guides/safety-best-practices#safety-identifiers).

          seed: This feature is in Beta. If specified, our system will make a best effort to
              sample deterministically, such that repeated requests with the same `seed` and
              parameters should return the same result. Determinism is not guaranteed, and you
              should refer to the `system_fingerprint` response parameter to monitor changes
              in the backend.

          service_tier: Specifies the processing type used for serving the request.

              - If set to 'auto', then the request will be processed with the service tier
                configured in the Project settings. Unless otherwise configured, the Project
                will use 'default'.
              - If set to 'default', then the request will be processed with the standard
                pricing and performance for the selected model.
              - If set to '[flex](https://platform.openai.com/docs/guides/flex-processing)' or
                '[priority](https://openai.com/api-priority-processing/)', then the request
                will be processed with the corresponding service tier.
              - When not set, the default behavior is 'auto'.

              When the `service_tier` parameter is set, the response body will include the
              `service_tier` value based on the processing mode actually used to serve the
              request. This response value may be different from the value set in the
              parameter.

          stop: Not supported with latest reasoning models `o3` and `o4-mini`.

              Up to 4 sequences where the API will stop generating further tokens. The
              returned text will not contain the stop sequence.

          store: Whether or not to store the output of this chat completion request for use in
              our [model distillation](https://platform.openai.com/docs/guides/distillation)
              or [evals](https://platform.openai.com/docs/guides/evals) products.

              Supports text and image inputs. Note: image inputs over 8MB will be dropped.

          temperature: What sampling temperature to use, between 0 and 2. Higher values like 0.8 will
              make the output more random, while lower values like 0.2 will make it more
              focused and deterministic. We generally recommend altering this or `top_p` but
              not both.

          tool_choice: Controls which (if any) tool is called by the model. `none` means the model will
              not call any tool and instead generates a message. `auto` means the model can
              pick between generating a message or calling one or more tools. `required` means
              the model must call one or more tools. Specifying a particular tool via
              `{"type": "function", "function": {"name": "my_function"}}` forces the model to
              call that tool.

              `none` is the default when no tools are present. `auto` is the default if tools
              are present.

          tools: A list of tools the model may call. You can provide either
              [custom tools](https://platform.openai.com/docs/guides/function-calling#custom-tools)
              or [function tools](https://platform.openai.com/docs/guides/function-calling).

          top_logprobs: An integer between 0 and 20 specifying the maximum number of most likely tokens
              to return at each token position, each with an associated log probability. In
              some cases, the number of returned tokens may be fewer than requested.
              `logprobs` must be set to `true` if this parameter is used.

          top_p: An alternative to sampling with temperature, called nucleus sampling, where the
              model considers the results of the tokens with top_p probability mass. So 0.1
              means only the tokens comprising the top 10% probability mass are considered.

              We generally recommend altering this or `temperature` but not both.

          user: This field is being replaced by `safety_identifier` and `prompt_cache_key`. Use
              `prompt_cache_key` instead to maintain caching optimizations. A stable
              identifier for your end-users. Used to boost cache hit rates by better bucketing
              similar requests and to help OpenAI detect and prevent abuse.
              [Learn more](https://platform.openai.com/docs/guides/safety-best-practices#safety-identifiers).

          verbosity: Constrains the verbosity of the model's response. Lower values will result in
              more concise responses, while higher values will result in more verbose
              responses. Currently supported values are `low`, `medium`, and `high`.

          web_search_options: This tool searches the web for relevant results to use in a response. Learn more
              about the
              [web search tool](https://platform.openai.com/docs/guides/tools-web-search?api-mode=chat).

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds"""

        openai_response = await self.client.chat.completions.create(
            messages=messages,
            model=self.model_name,
            stream=False,
            stream_options=None,
            **kwargs,
        )

        return OAIResponse(openai_response)
    
    async def aclose(self) -> None:
        """Close the client and any resources it holds."""
        if self._owns_client and self.client is not None:
            await self.client.close()
            self.client = None
