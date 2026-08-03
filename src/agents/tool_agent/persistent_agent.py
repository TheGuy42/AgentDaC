from __future__ import annotations

from enum import StrEnum

from src.agents.base import BaseAgent
from src.agents.tool_agent.parsing import NativeParser
from src.agents.tool_agent.schemas import GuidedTools
from src.aliases import Message, UserMessage, ToolMessage
from src.configs import PromptConfig, DecompConfig, ToolSpecs
from src.inference import InferenceClient, InferenceResponse
from src.trajectory import Trajectory
from src.utils.logging import create_logger


logger = create_logger(__name__)


METRIC_PREFIXES = ("direct", "subtree")
METRIC_COUNTERS = ("calls", "tasks", "thinks", "agents", "chats")


class ToolPersistentActions(StrEnum):
    CREATE_NEW_SUB_AGENT = "create_new_sub_agent"
    MESSAGE_CURRENT_SUB_AGENT = "send_message_to_current_sub_agent"


class ToolPersistentErrors(StrEnum):
    CLIENT_ERROR = "client_error"
    PARSE_ERROR = "parse_error"
    TOOL_UNAVAILABLE = "tool_unavailable"
    OUT_OF_ROUNDS = "out_of_rounds"


class ToolPersistentAgent(BaseAgent):
    @classmethod
    def error_kinds(cls) -> type[StrEnum]:
        return ToolPersistentErrors

    def __init__(
        self,
        client: InferenceClient,
        prompt_config: PromptConfig,
        decomp_config: DecompConfig,
        tool_parser: NativeParser,
        current_depth: int = 0,
        additional_histories: bool = False,
        verbose: bool = False,
    ) -> None:
        super().__init__(
            client=client,
            prompt_config=prompt_config,
            decomp_config=decomp_config,
            current_depth=current_depth,
            additional_histories=additional_histories,
            verbose=verbose,
        )

        self.tool_parser = tool_parser
        self.guided_tools = self._create_tools()
        self.trajectory.tools = self.guided_tools.build()

        self.metrics.update({f"{prefix}_{counter}": 0 for counter in METRIC_COUNTERS for prefix in METRIC_PREFIXES})
        self.metrics.update({"subtree_depth": 0, "direct_tokens": 0})

        # We support a persistent sub-agent across chat rounds
        self.sub_agent: ToolPersistentAgent | None = None

    def _create_tools(self) -> GuidedTools:
        specs = self.prompt_config.tools
        if not isinstance(specs, ToolSpecs):
            raise ValueError(f"Expected ToolSpecs, got {type(specs).__name__}")

        if self.decomp_config.is_leaf(self.current_depth):
            return GuidedTools(specs)  # leaf: no delegation, answers directly
        return GuidedTools(specs, *ToolPersistentActions)

    async def call(self, messages: list[Message], **kwargs) -> InferenceResponse:
        kwargs = self.client.update_kwargs(kwargs, include_stop_str_in_output=True)
        return await super().call(
            messages,
            tools=self.guided_tools.build() or None,
            stop=self.tool_parser.stop_tag,
            **kwargs,
        )

    def create_subagent(self) -> ToolPersistentAgent:
        agent = ToolPersistentAgent(
            client=self.client,
            prompt_config=self.prompt_config,
            decomp_config=self.decomp_config,
            current_depth=self.current_depth + 1,
            tool_parser=self.tool_parser,
            additional_histories=False,
            verbose=self.verbose,
        )

        if self.additional_histories:
            self.trajectory.histories.append(agent.trajectory)

        return agent

    def allowed_actions(self) -> list[str]:
        DC = self.decomp_config
        allowed = []
        if (not DC.is_leaf(self.current_depth)) and DC.has_tasks():
            allowed.append(ToolPersistentActions.CREATE_NEW_SUB_AGENT)
            if self.sub_agent is not None:
                allowed.append(ToolPersistentActions.MESSAGE_CURRENT_SUB_AGENT)
        return allowed

    def status_message(self) -> UserMessage:
        DC = self.decomp_config
        tool_names = [self.guided_tools.name_of(action) for action in self.allowed_actions()]
        content = (
            f"<controller_state>\n"
            f"remaining_rounds: {max(DC.max_rounds - DC.total_rounds, 0)}\n"
            f"remaining_delegations: {max(DC.max_tasks - DC.total_tasks, 0)}\n"
            f"allowed_tools: {tool_names if tool_names else 'None'}\n"
            f"</controller_state>"
        )

        return UserMessage(role="user", content=content)

    async def chat(self, prompt: Message, **kwargs) -> Trajectory:
        if prompt.get("role") != "user":
            logger.warning(f"Prompt role is expected to be 'user', but got {prompt.get('role')}.")

        self.decomp_config.reset()
        self.append_message(prompt)

        # Reset metrics of the run
        for k in self.metrics.keys():
            if any(k.startswith(prefix) for prefix in METRIC_PREFIXES):
                self.metrics[k] = 0

        for prefix in METRIC_PREFIXES:
            self.metrics[f"{prefix}_chats"] += 1

        while True:
            # Status message, only if we are not at a leaf (tools are available)
            if self.guided_tools.actions:
                status_message = self.status_message()
                self.append_message(status_message)

            for prefix in METRIC_PREFIXES:
                self.metrics[f"{prefix}_calls"] += 1

            try:
                # Model turn
                completion = await self.call(self.trajectory.messages(), **kwargs)
            except Exception as e:
                logger.debug(f"Error during model call: {e}")
                self.trajectory.error(kind=ToolPersistentErrors.CLIENT_ERROR, message=str(e))
                return self.trajectory.finish()

            self.append_message(completion)

            if completion.total_tokens is not None:
                self.metrics["direct_tokens"] = completion.total_tokens

            try:
                # Parse reasoning and tool calls from the model's output
                turn = self.tool_parser.parse(completion)
            except Exception as e:
                logger.debug(f"Failed to parse model output: {e}")
                self.trajectory.error(kind=ToolPersistentErrors.PARSE_ERROR, message=str(e))
                return self.trajectory.finish()

            if turn.reasoning:
                for prefix in METRIC_PREFIXES:
                    self.metrics[f"{prefix}_thinks"] += 1

            # Terminal: model chose to answer
            if turn.tool_call is None:
                return self.trajectory.finish()

            # Terminal: out of rounds
            if not self.decomp_config.has_rounds():
                error_text = "[error] Model ran out of rounds."
                self.trajectory.error(kind=ToolPersistentErrors.OUT_OF_ROUNDS, message=error_text)
                return self.trajectory.finish()

            try:  # Parse tool call argument
                call = self.guided_tools.parse(turn.tool_call, repair=False)

            except Exception as e:
                error_text = f"[error] Failed to parse tool call: {e}"
                self.trajectory.error(kind=ToolPersistentErrors.PARSE_ERROR, message=error_text)
                error_message = ToolMessage(role="tool", content=error_text, tool_call_id=turn.tool_call.id)
                self.append_message(error_message)
                self.decomp_config.update_round(num_tasks=0)
                continue

            # Make sure the tool is available in the current state
            if call.action not in self.allowed_actions():
                error_text = f"[error] Tool `{turn.tool_call.function.name}` is unavailable in the current state."
                self.trajectory.error(kind=ToolPersistentErrors.TOOL_UNAVAILABLE, message=error_text)
                error_message = ToolMessage(role="tool", content=error_text, tool_call_id=turn.tool_call.id)
                self.append_message(error_message)
                self.decomp_config.update_round(num_tasks=0)
                continue

            # Create a new sub-agent
            if call.action == ToolPersistentActions.CREATE_NEW_SUB_AGENT or self.sub_agent is None:
                self.sub_agent = self.create_subagent()
                for prefix in METRIC_PREFIXES:
                    self.metrics[f"{prefix}_agents"] += 1

            # Issue a sub-task to the current sub-agent
            if call.action in (ToolPersistentActions.CREATE_NEW_SUB_AGENT, ToolPersistentActions.MESSAGE_CURRENT_SUB_AGENT):
                task = UserMessage(role="user", content=call.args["text"])
                task_answer = await self.sub_agent.answer(task, **kwargs)
                if task_answer is None:
                    task_answer = "[error] sub-agent failed to produce an answer."

                task_response = ToolMessage(role="tool", content=task_answer, tool_call_id=turn.tool_call.id)
                self.append_message(task_response)

                for prefix in METRIC_PREFIXES:
                    self.metrics[f"{prefix}_tasks"] += 1

                # Fold in the sub-agent's subtree contribution from this single invocation.
                for q in METRIC_COUNTERS:
                    self.metrics[f"subtree_{q}"] += self.sub_agent.metrics[f"subtree_{q}"]

                child_depth = 1 + self.sub_agent.metrics["subtree_depth"]
                self.metrics["subtree_depth"] = max(self.metrics["subtree_depth"], child_depth)

                self.decomp_config.update_round(num_tasks=1)

            else:
                # unknown tool name, should be impossible since already handled above
                raise ValueError(f"Unhandled tool call name: {turn.tool_call.function.name}.")

    def parse_answer(self, message: Message | InferenceResponse) -> str | None:
        if not isinstance(message, InferenceResponse):
            logger.debug(f"Expected an InferenceResponse, got {type(message)}")
            return None

        try:
            parsed_content = self.tool_parser.parse(message).content
            return parsed_content.strip() if parsed_content is not None else None

        except Exception as e:
            logger.debug(f"Failed to parse answer from message: {e}")
            return None
