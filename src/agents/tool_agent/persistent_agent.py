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
        if len(available_tools) > 0:
            options = [f"{schema['function']['name']}" for schema in available_tools]
            content = (
                f"[status] rounds remaining: {max(self.decomp_config.max_rounds - self.decomp_config.total_rounds, 0)}; "
                f"sub-tasks remaining: {max(self.decomp_config.max_tasks - self.decomp_config.total_tasks, 0)}. "
                f"Available tools: {options}, or write your final answer directly to finish."
            )
        else:
            content = "[status] no delegations remaining. Write your final answer directly now."

        return UserMessage(role="user", content=content)

    async def chat(
        self,
        prompt: Message,
        verbose: bool = False,
        **kwargs,
    ) -> Trajectory:
        if prompt.get("role") != "user":
            logger.warning(f"Prompt role is expected to be 'user', but got {prompt.get('role')}.")

        self.decomp_config.reset()
        self.trajectory.messages_and_responses.append(prompt)

        if verbose:
            print(trajectory_string(self.trajectory, indent=self.current_depth))

        # Reset metrics of the run
        for k in self.metrics.keys():
            if any(k.startswith(prefix) for prefix in METRIC_PREFIXES):
                self.metrics[k] = 0

        for prefix in METRIC_PREFIXES:
            self.metrics[f"{prefix}_chats"] += 1

        while True:
            can_delegate = self._can_delegate()

            # Status message, only if we are not at a leaf (tools are available)
            if len(self.tool_map) > 0:
                status_message = self._status_message()
                self.trajectory.messages_and_responses.append(status_message)

                if verbose:
                    print(message_string(status_message, indent=self.current_depth))

            # Model turn
            completion = await self._call(self.trajectory.messages(), **kwargs)
            self.trajectory.messages_and_responses.append(completion)

            for prefix in METRIC_PREFIXES:
                self.metrics[f"{prefix}_calls"] += 1
            if completion.total_tokens is not None:
                self.metrics["direct_tokens"] = completion.total_tokens

            if verbose:
                print(message_string(self.trajectory.messages()[-1], indent=self.current_depth))

            # Parse reasoning and tool calls from the model's output
            turn = self.tool_parser.parse(completion.content)

            if turn.reasoning:
                for prefix in METRIC_PREFIXES:
                    self.metrics[f"{prefix}_thinks"] += 1

            available_names = [schema["function"]["name"] for schema in self._available_tools()]

            # Terminal: no too call (or no budget)
            if turn.tool_call is None or turn.tool_call.function.name not in available_names or not can_delegate:
                self.decomp_config.update_round(num_tasks=0)
                break

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
                    # Feed error message to the model
                    error_text = f"`{arg_name}` should appear as an argument of the tool call: {turn.tool_call.function.name}"
                    logger.warning(error_text)
                    error_message = UserMessage(role="user", content=f"[error] {error_text}. Please try again.")
                    self.trajectory.messages_and_responses.append(error_message)
                    self.decomp_config.update_round(num_tasks=0)
                    continue

                task = UserMessage(role="user", content=task_content)
                task_answer = await self.sub_agent.answer(task, verbose, **kwargs)
                task_response = ToolMessage(role="tool", content=task_answer, tool_call_id=turn.tool_call.id)
                self.trajectory.messages_and_responses.append(task_response)

                if verbose:
                    print(message_string(self.trajectory.messages()[-1], indent=self.current_depth))

                for prefix in METRIC_PREFIXES:
                    self.metrics[f"{prefix}_tasks"] += 1

                # Fold in the sub-agent's subtree contribution from this single invocation.
                for q in METRIC_COUNTERS:
                    self.metrics[f"subtree_{q}"] += self.sub_agent.metrics[f"subtree_{q}"]

                child_depth = 1 + self.sub_agent.metrics["subtree_depth"]
                self.metrics["subtree_depth"] = max(self.metrics["subtree_depth"], child_depth)

                self.decomp_config.update_round(num_tasks=1)

            else:
                # Feed error message to the model
                error_text = f"Unexpected tool call name: {turn.tool_call.function.name}"
                logger.warning(error_text)
                error_message = UserMessage(role="user", content=f"[error] {error_text}. Please try again.")
                self.trajectory.messages_and_responses.append(error_message)
                self.decomp_config.update_round(num_tasks=0)

        # Update final stats
        completed = int((completion.finish_reason != "length") and turn.content is not None)
        incomplete = 1 - completed

        # This agent's own response completion, counted across all four quadrants
        for prefix in METRIC_PREFIXES:
            self.metrics[f"{prefix}_responses_completed"] += completed
            self.metrics[f"{prefix}_responses_incomplete"] += incomplete

        self.trajectory.finish()
        return self.trajectory

    def parse_answer(self, message: Message) -> str:
        if message["role"] != "assistant":
            logger.error(f"Expected message role 'assistant', got {message['role']}")
            raise ValueError("Message role must be 'assistant' to extract answer.")

        content = message.get("content")
        if not isinstance(content, str):
            logger.error(f"Expected message content to be a string, got {type(content)}")
            raise ValueError("Message content must be a string.")

        parsed_content = self.tool_parser.parse(content).content
        if parsed_content is None:
            logger.warning("Parsed content is None; returning empty string as answer.")
            return ""

        return parsed_content.strip()
