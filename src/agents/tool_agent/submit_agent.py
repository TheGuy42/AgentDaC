from __future__ import annotations

from src.agents.tool_agent.persistent_agent import (
    ToolPersistentAgent,
    ToolPersistentErrors,
    CREATE_NEW_SUB_AGENT,
    MESSAGE_CURRENT_SUB_AGENT,
    METRIC_PREFIXES,
    METRIC_COUNTERS,
)

from src.agents.tool_agent.schemas import tool_schema
from src.aliases import Message, UserMessage, ToolMessage, ToolSchema
from src.inference import InferenceResponse
from src.trajectory import Trajectory
from src.utils.logging import create_logger


logger = create_logger(__name__)


SUBMIT_ANSWER = "submit_answer"


class ToolSubmitAgent(ToolPersistentAgent):
    """A persistent tool agent that finishes by calling `submit_answer`.

    Identical to `ToolPersistentAgent`, except the final answer is delivered as the
    argument of a `submit_answer` tool call instead of as free-form message content.
    """

    def _build_tools(self) -> dict[str, ToolSchema]:
        tools = super()._build_tools()
        tools[SUBMIT_ANSWER] = tool_schema(
            name=SUBMIT_ANSWER,
            desc=(
                "Submit your final answer and finish the task. Call this exactly once, "
                "when you are done. The `answer` must be the final answer only, with no "
                "extra commentary, in the exact format the task requests."
            ),
            arg_name="answer",
            arg_desc=("The final answer only, in the exact requested format (e.g. a single UCI move like e2e4)."),
        )
        return tools

    def available_tools(self) -> list[ToolSchema]:
        available = super().available_tools()
        available.append(self.tool_map[SUBMIT_ANSWER])
        return available

    def create_subagent(self) -> ToolSubmitAgent:
        agent = ToolSubmitAgent(
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
                status_message = self.status_message()
                self.append_message(status_message)

            for prefix in METRIC_PREFIXES:
                self.metrics[f"{prefix}_calls"] += 1

            try:
                # Model turn
                completion = await self.call(self.trajectory.messages(), **kwargs)
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

            # Error: no tool call
            if turn.tool_call is None:
                error_text = "[error] Model output did not contain a tool call."
                self.trajectory.error(kind=ToolPersistentErrors.PARSE_ERROR, message=error_text)
                if not self.decomp_config.has_rounds():
                    return self.trajectory.finish()
                error_message = UserMessage(role="user", content=error_text)
                self.append_message(error_message)
                self.decomp_config.update_round(num_tasks=0)
                continue

            try:  # Parse tool call argument
                args = self._parse_arguments(turn.tool_call, repair=False)

            except Exception as e:
                error_text = f"[error] Failed to parse tool call: {e}"
                self.trajectory.error(kind=ToolPersistentErrors.PARSE_ERROR, message=error_text)
                if not self.decomp_config.has_rounds():
                    return self.trajectory.finish()
                error_message = ToolMessage(role="tool", content=error_text, tool_call_id=turn.tool_call.id)
                self.append_message(error_message)
                self.decomp_config.update_round(num_tasks=0)
                continue

            # Make sure the tool is available in the current state
            available_names = [schema["function"]["name"] for schema in self.available_tools()]

            if turn.tool_call.function.name not in available_names:
                error_text = f"[error] Tool `{turn.tool_call.function.name}` is unavailable in the current state."
                self.trajectory.error(kind=ToolPersistentErrors.TOOL_UNAVAILABLE, message=error_text)
                if not self.decomp_config.has_rounds():
                    return self.trajectory.finish()
                error_message = ToolMessage(role="tool", content=error_text, tool_call_id=turn.tool_call.id)
                self.append_message(error_message)
                self.decomp_config.update_round(num_tasks=0)
                continue

            # Terminal: the model submitted an answer, we are done
            if turn.tool_call.function.name == SUBMIT_ANSWER:
                self.decomp_config.update_round(num_tasks=0)
                return self.trajectory.finish()

            # Terminal: out of rounds, we use last message as the final answer
            if not self.decomp_config.has_rounds():
                error_text = "[error] Model ran out of rounds."
                self.trajectory.error(kind=ToolPersistentErrors.OUT_OF_ROUNDS, message=error_text)
                return self.trajectory.finish()

            # Create a new sub-agent
            if turn.tool_call.function.name == CREATE_NEW_SUB_AGENT or self.sub_agent is None:
                self.sub_agent = self.create_subagent()
                for prefix in METRIC_PREFIXES:
                    self.metrics[f"{prefix}_agents"] += 1

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

        try:
            turn = self.tool_parser.parse(message)
            if turn.tool_call is None or turn.tool_call.function.name != SUBMIT_ANSWER:
                return None

            args = self._parse_arguments(turn.tool_call, repair=True)
            return args["answer"].strip()

        except Exception as e:
            logger.error(f"Failed to parse answer from tool call: {e}")
            return None
