from __future__ import annotations

import json
from enum import StrEnum
from vllm.entrypoints.openai.engine.protocol import ToolCall
import json_repair

from src.agents.base import BaseAgent
from src.agents.tool_agent.parsing import NativeToolParser
from src.agents.tool_agent.schemas import tool_schema
from src.aliases import Message, UserMessage, ToolMessage, ToolSchema
from src.configs import PromptConfig, DecompConfig
from src.inference import InferenceClient, InferenceResponse
from src.trajectory import Trajectory
from src.utils.logging import create_logger


logger = create_logger(__name__)


CREATE_NEW_SUB_AGENT = "create_new_sub_agent"
MESSAGE_CURRENT_SUB_AGENT = "send_message_to_current_sub_agent"

METRIC_PREFIXES = ("direct", "subtree")
METRIC_COUNTERS = ("calls", "tasks", "thinks", "agents", "chats")


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
        tool_parser: NativeToolParser,
        current_depth: int = 0,
        additional_histories: bool = False,
        verbose: bool = False,
    ) -> None:
        super().__init__(client, prompt_config, decomp_config, current_depth, additional_histories, verbose)

        self.tool_parser = tool_parser
        self.tool_map = self.build_tools()
        self.trajectory.tools = list(self.tool_map.values())

        self.metrics.update({f"{prefix}_{counter}": 0 for counter in METRIC_COUNTERS for prefix in METRIC_PREFIXES})
        self.metrics.update({"subtree_depth": 0, "direct_tokens": 0})

        # We support a persistent sub-agent across chat rounds
        self.sub_agent: ToolPersistentAgent | None = None

    def build_tools(self) -> dict[str, ToolSchema]:
        if self.decomp_config.is_leaf(self.current_depth):
            return {}  # leaf: no delegation, answers directly

        return {
            CREATE_NEW_SUB_AGENT: tool_schema(
                name=CREATE_NEW_SUB_AGENT,
                desc=(
                    "Create a fresh sub-agent (discarding any previous one) and give it this sub-task. "
                    "The sub-agent starts with NO memory or context of this conversation, so the `text` must "
                    "be fully self-contained: restate the complete position and everything needed to solve it."
                ),
                arg_name="text",
                arg_desc=(
                    "A clear, fully self-contained description of the sub-task, including the complete "
                    "position/board and the exact answer format required."
                ),
            ),
            MESSAGE_CURRENT_SUB_AGENT: tool_schema(
                name=MESSAGE_CURRENT_SUB_AGENT,
                desc=(
                    "Continue with the current sub-agent (which remembers its earlier work) on a follow-up "
                    "sub-task. Phrase it in terms of what it has already done."
                ),
                arg_name="text",
                arg_desc="A clear description of the follow-up sub-task for the current sub-agent.",
            ),
        }

    async def _call(self, messages: list[Message], **kwargs) -> InferenceResponse:
        # Cap the turn at a single tool call by stopping at its closing tag.
        stops = list(kwargs.get("stop") or [])
        if self.tool_parser.stop_tag not in stops:
            stops.append(self.tool_parser.stop_tag)
        kwargs["stop"] = stops
        kwargs = self.client.update_kwargs(kwargs, include_stop_str_in_output=True)
        return await super()._call(messages, tools=list(self.tool_map.values()) or None, **kwargs)

    def _create_subagent(self) -> ToolPersistentAgent:
        return ToolPersistentAgent(
            client=self.client,
            prompt_config=self.prompt_config,
            decomp_config=self.decomp_config,
            current_depth=self.current_depth + 1,
            tool_parser=self.tool_parser,
            additional_histories=False,
            verbose=self.verbose,
        )

    def _available_tools(self) -> list[ToolSchema]:
        available = []
        DC = self.decomp_config
        if (not DC.is_leaf(self.current_depth)) and DC.has_tasks():
            available += [self.tool_map[CREATE_NEW_SUB_AGENT], self.tool_map[MESSAGE_CURRENT_SUB_AGENT]]
        return available

    def _status_message(self) -> UserMessage:
        DC = self.decomp_config
        available_tools = self._available_tools()
        action_names = [f"{schema['function']['name']}" for schema in available_tools] + ["final answer (no tool)"]
        content = (
            f"<controller_state>\n"
            f"remaining_rounds: {max(DC.max_rounds - DC.total_rounds, 0)}\n"
            f"remaining_delegations: {max(DC.max_tasks - DC.total_tasks, 0)}\n"
            f"allowed_actions: {action_names}\n"
            f"</controller_state>"
        )

        return UserMessage(role="user", content=content)

    def _parse_arguments(self, tool_call: ToolCall, repair: bool = True) -> dict[str, str]:
        """
        Parse the tool call arguments and return the argument name and content.

        Args:
            tool_call (ToolCall): The tool call object containing the function name and arguments.
            repair (bool): Whether to attempt to repair malformed JSON arguments.

        Returns:
            dict[str, str]: A dictionary containing the argument name and content.

        Raises:
            ValueError: If the tool call type is not "function", if the tool name is unknown, if the arguments cannot be parsed, or if required arguments are missing.
        """

        if tool_call.type != "function":
            raise ValueError(f"Expected a function tool call, got type: {tool_call.type}")

        tool_schema = self.tool_map.get(tool_call.function.name)
        if tool_schema is None:
            raise ValueError(f"Unknown tool call name: {tool_call.function.name}.")

        try:
            call_args = json.loads(tool_call.function.arguments)

        except json.JSONDecodeError as e:
            if not repair:
                raise ValueError(f"Failed to parse tool call arguments: {e}")

            call_args = json_repair.loads(
                tool_call.function.arguments,
                schema=tool_schema["function"].get("parameters", {}),
                skip_json_loads=True,
                schema_repair_mode="standard",
            )

        if not isinstance(call_args, dict):
            raise ValueError(f"Parsed tool call arguments are not a dictionary, got {type(call_args)}")

        arg_names: list[str] = list(tool_schema["function"].get("parameters", {}).get("properties", {}).keys())  # type: ignore

        if not all(arg_name in call_args for arg_name in arg_names):
            missing_args = [arg_name for arg_name in arg_names if arg_name not in call_args]
            raise ValueError(f"Missing required argument(s) {missing_args} for tool call: {tool_call.function.name}.")

        if not all(isinstance(call_args[arg_name], str) for arg_name in arg_names):
            raise ValueError(f"Argument(s) for tool call: {tool_call.function.name} are not strings.")

        return {arg_name: call_args[arg_name] for arg_name in arg_names}

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
            if len(self.tool_map) > 0:
                status_message = self._status_message()
                self.append_message(status_message)

            for prefix in METRIC_PREFIXES:
                self.metrics[f"{prefix}_calls"] += 1

            try:
                # Model turn
                completion = await self._call(self.trajectory.messages(), **kwargs)
            except Exception as e:
                logger.error(f"Error during model call: {e}")
                self.trajectory.error(kind=ToolPersistentErrors.CLIENT_ERROR, message=str(e))
                return self.trajectory.finish()

            self.append_message(completion)

            if completion.total_tokens is not None:
                self.metrics["direct_tokens"] = completion.total_tokens

            try:
                # Parse reasoning and tool calls from the model's output
                turn = self.tool_parser.parse(completion)
            except Exception as e:
                logger.warning(f"Failed to parse model output: {e}")
                self.trajectory.error(kind=ToolPersistentErrors.PARSE_ERROR, message=str(e))
                return self.trajectory.finish()

            if turn.reasoning:
                for prefix in METRIC_PREFIXES:
                    self.metrics[f"{prefix}_thinks"] += 1

            # Terminal: model chose to answer
            if turn.tool_call is None:
                self.decomp_config.update_round(num_tasks=0)
                return self.trajectory.finish()

            # Terminal: out of rounds, use last message as the final answer
            if not self.decomp_config.has_rounds():
                error_text = "[error] Model ran out of rounds."
                self.trajectory.error(kind=ToolPersistentErrors.OUT_OF_ROUNDS, message=error_text)
                return self.trajectory.finish()

            try:  # Parse tool call argument
                args = self._parse_arguments(turn.tool_call, repair=True)

            except Exception as e:
                error_text = f"[error] Failed to parse tool call: {e}"
                self.trajectory.error(kind=ToolPersistentErrors.PARSE_ERROR, message=error_text)
                error_message = ToolMessage(role="tool", content=error_text, tool_call_id=turn.tool_call.id)
                self.append_message(error_message)
                self.decomp_config.update_round(num_tasks=0)
                continue

            # Make sure the tool is available in the current state
            available_names = [schema["function"]["name"] for schema in self._available_tools()]

            if turn.tool_call.function.name not in available_names:
                error_text = f"[error] Tool `{turn.tool_call.function.name}` is unavailable in the current state."
                self.trajectory.error(kind=ToolPersistentErrors.TOOL_UNAVAILABLE, message=error_text)
                error_message = ToolMessage(role="tool", content=error_text, tool_call_id=turn.tool_call.id)
                self.append_message(error_message)
                self.decomp_config.update_round(num_tasks=0)
                continue

            # Create a new sub-agent
            if turn.tool_call.function.name == CREATE_NEW_SUB_AGENT or self.sub_agent is None:
                self.sub_agent = self._create_subagent()
                for prefix in METRIC_PREFIXES:
                    self.metrics[f"{prefix}_agents"] += 1

                if self.additional_histories:
                    self.trajectory.histories.append(self.sub_agent.trajectory)

            # Issue a sub-task to the current sub-agent
            if turn.tool_call.function.name in (CREATE_NEW_SUB_AGENT, MESSAGE_CURRENT_SUB_AGENT):
                task = UserMessage(role="user", content=args["text"])
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
            logger.error(f"Expected an InferenceResponse, got {type(message)}")
            return None

        parsed_content = self.tool_parser.parse(message).content
        return parsed_content.strip() if parsed_content is not None else None
