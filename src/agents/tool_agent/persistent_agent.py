from __future__ import annotations

import json

from src.agents.base import BaseAgent
from src.agents.tool_agent.parsing import NativeToolParser
from src.agents.tool_agent.schemas import tool_schema
from src.aliases import Message, UserMessage, ToolMessage, ToolSchema
from src.configs import PromptConfig, DecompConfig
from src.inference import InferenceClient, InferenceResponse
from src.trajectory import Trajectory
from src.utils.logging import create_logger
from src.utils.visualize import trajectory_string, message_string


logger = create_logger(__name__)


CREATE_NEW_SUB_AGENT = "create_new_sub_agent"
MESSAGE_CURRENT_SUB_AGENT = "send_message_to_current_sub_agent"
DELEGATION_TOOLS = (CREATE_NEW_SUB_AGENT, MESSAGE_CURRENT_SUB_AGENT)

METRIC_PREFIXES = ("direct", "subtree")
METRIC_COUNTERS = ("calls", "tasks", "thinks", "agents", "chats", "responses_completed", "responses_incomplete")


class ToolPersistentAgent(BaseAgent):
    def __init__(
        self,
        client: InferenceClient,
        prompt_config: PromptConfig,
        decomp_config: DecompConfig,
        tool_parser: NativeToolParser,
        current_depth: int = 0,
        additional_histories: bool = False,
    ) -> None:
        super().__init__(client, prompt_config, decomp_config, current_depth, additional_histories)

        self.tool_parser = tool_parser
        self.tool_map = self.build_tools()
        self.trajectory.tools = list(self.tool_map.values())

        self.metrics.update({f"{prefix}_{counter}": 0 for counter in METRIC_COUNTERS for prefix in METRIC_PREFIXES})
        self.metrics.update({"subtree_depth": 0, "direct_tokens": 0})

        # We support a persistent sub-agent across chat rounds
        self.sub_agent: ToolPersistentAgent | None = None

    def build_tools(self) -> dict[str, ToolSchema]:
        if self.current_depth >= self.decomp_config.max_depth:
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

    def _can_delegate(self) -> bool:
        return (
            (self.current_depth < self.decomp_config.max_depth)
            and (self.decomp_config.total_rounds < self.decomp_config.max_rounds)
            and (self.decomp_config.total_tasks < self.decomp_config.max_tasks)
        )

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
        )

    def _available_tools(self) -> list[ToolSchema]:
        available = []
        if self._can_delegate():
            available += [self.tool_map[CREATE_NEW_SUB_AGENT], self.tool_map[MESSAGE_CURRENT_SUB_AGENT]]
        return available

    def _status_message(self) -> UserMessage:
        available_tools = self._available_tools()
        action_names = [f"{schema['function']['name']}" for schema in available_tools] + ["final answer (no tool)"]
        content = (
            f"<controller_state>\n"
            f"remaining_rounds: {max(self.decomp_config.max_rounds - self.decomp_config.total_rounds, 0)}\n"
            f"remaining_delegations: {max(self.decomp_config.max_tasks - self.decomp_config.total_tasks, 0)}\n"
            f"allowed_actions: {action_names}\n"
            f"</controller_state>"
        )

        return UserMessage(role="user", content=content)

    def _append_message(self, message: Message | InferenceResponse, verbose: bool = False):
        self.trajectory.messages_and_responses.append(message)
        if verbose:
            if isinstance(message, InferenceResponse):
                message = self.trajectory.messages()[-1]  # Let trajectory.messages() handle the conversion
            print(message_string(message, indent=self.current_depth))

    async def chat(
        self,
        prompt: Message,
        verbose: bool = False,
        **kwargs,
    ) -> Trajectory:
        if prompt.get("role") != "user":
            logger.warning(f"Prompt role is expected to be 'user', but got {prompt.get('role')}.")

        self.decomp_config.reset()
        self._append_message(prompt, verbose=False)

        if verbose:
            print(trajectory_string(self.trajectory, indent=self.current_depth))

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
                self._append_message(status_message, verbose=verbose)

            # Model turn
            completion = await self._call(self.trajectory.messages(), **kwargs)
            self._append_message(completion, verbose=verbose)

            for prefix in METRIC_PREFIXES:
                self.metrics[f"{prefix}_calls"] += 1
            if completion.total_tokens is not None:
                self.metrics["direct_tokens"] = completion.total_tokens

            if verbose:
                print(message_string(self.trajectory.messages()[-1], indent=self.current_depth))

            # Parse reasoning and tool calls from the model's output
            turn = self.tool_parser.parse(completion)

            if turn.reasoning:
                for prefix in METRIC_PREFIXES:
                    self.metrics[f"{prefix}_thinks"] += 1

            available_names = [schema["function"]["name"] for schema in self._available_tools()]

            # Terminal: no too call
            if turn.tool_call is None:
                self.decomp_config.update_round(num_tasks=0)
                break

            # Terminal: no available tools, but the model tried to call one
            if turn.tool_call is not None and len(available_names) == 0:
                logger.info("Model attempted to call a tool, but no tools are available.")
                break  # We will still try to extract final answer from the turn.content of this turn

            # Illegal tool name
            if turn.tool_call.function.name not in self.tool_map.keys():
                error_message = ToolMessage(
                    role="tool",
                    content=f"[error] Unexpected tool call name: {turn.tool_call.function.name}.",
                    tool_call_id=turn.tool_call.id,
                )
                self._append_message(error_message, verbose=verbose)
                self.decomp_config.update_round(num_tasks=0)
                continue

            # Legal tool name, but its not available
            if turn.tool_call.function.name in self.tool_map.keys() and turn.tool_call.function.name not in available_names:
                error_message = ToolMessage(
                    role="tool",
                    content=f"[error] `{turn.tool_call.function.name}` unavailable in the current state.",
                    tool_call_id=turn.tool_call.id,
                )
                self._append_message(error_message, verbose=verbose)
                self.decomp_config.update_round(num_tasks=0)
                continue

            # Create a new sub-agent
            if turn.tool_call.function.name == CREATE_NEW_SUB_AGENT or self.sub_agent is None:
                self.sub_agent = self._create_subagent()
                for prefix in METRIC_PREFIXES:
                    self.metrics[f"{prefix}_agents"] += 1

                if self.additional_histories:
                    # Each sub-agent defines its own history
                    # The history is dynamically updated as the sub-agent is invoked, as expected.
                    self.trajectory.histories.append(self.sub_agent.trajectory)

            # Issue a sub-task to the current sub-agent
            if turn.tool_call.function.name in (CREATE_NEW_SUB_AGENT, MESSAGE_CURRENT_SUB_AGENT):
                # Extract the tool call argument content
                call_args: dict = json.loads(turn.tool_call.function.arguments or "{}")
                arg_names = list(self.tool_map[turn.tool_call.function.name]["function"].get("parameters", {}).get("properties", {}).keys())  # type: ignore
                arg_name = arg_names[0] if len(arg_names) > 0 else None

                task_content = call_args.get(arg_name, None)
                if task_content is None and len(call_args) == 1:
                    # backup: if there is only a single arg then simply use it
                    task_content = list(call_args.values())[0]

                if task_content is None:
                    error_message = ToolMessage(
                        role="tool",
                        content=f"[error]`{arg_name}` should appear as an argument of the tool call: {turn.tool_call.function.name}.",
                        tool_call_id=turn.tool_call.id,
                    )
                    self._append_message(error_message, verbose=verbose)
                    self.decomp_config.update_round(num_tasks=0)
                    continue

                task = UserMessage(role="user", content=task_content)
                task_answer = await self.sub_agent.answer(task, verbose, **kwargs)
                task_response = ToolMessage(role="tool", content=task_answer, tool_call_id=turn.tool_call.id)
                self._append_message(task_response, verbose=verbose)

                for prefix in METRIC_PREFIXES:
                    self.metrics[f"{prefix}_tasks"] += 1

                # Fold in the sub-agent's subtree contribution from this single invocation.
                for q in METRIC_COUNTERS:
                    self.metrics[f"subtree_{q}"] += self.sub_agent.metrics[f"subtree_{q}"]

                child_depth = 1 + self.sub_agent.metrics["subtree_depth"]
                self.metrics["subtree_depth"] = max(self.metrics["subtree_depth"], child_depth)

                self.decomp_config.update_round(num_tasks=1)

            else:
                error_message = ToolMessage(
                    role="tool",
                    content=f"[error] Unexpected tool call name: {turn.tool_call.function.name}.",
                    tool_call_id=turn.tool_call.id,
                )
                self._append_message(error_message, verbose=verbose)
                self.decomp_config.update_round(num_tasks=0)
                continue

        # Update final stats
        completed = int((completion.finish_reason != "length") and (turn.tool_call is None) and (turn.content is not None))
        incomplete = 1 - completed

        # This agent's own response completion, counted across all four quadrants
        for prefix in METRIC_PREFIXES:
            self.metrics[f"{prefix}_responses_completed"] += completed
            self.metrics[f"{prefix}_responses_incomplete"] += incomplete

        self.trajectory.finish()
        return self.trajectory

    def parse_answer(self, message: Message | InferenceResponse) -> str:
        if not isinstance(message, InferenceResponse):
            logger.error(f"Expected an InferenceResponse, got {type(message)}")
            raise ValueError("parse_answer expects an InferenceResponse.")

        parsed_content = self.tool_parser.parse(message).content
        if parsed_content is None:
            logger.warning("Parsed content is None; returning empty string as answer.")
            return ""

        return parsed_content.strip()
