from __future__ import annotations
import json
from dataclasses import dataclass

import json_repair
from openai.types.shared_params.function_definition import FunctionDefinition
from vllm.entrypoints.openai.engine.protocol import ToolCall

from src.aliases import ToolSchema
from src.configs.prompt_config import ToolSpecs


@dataclass
class ToolTurn:
    action: str
    args: dict[str, str]


class GuidedTools:
    def __init__(self, specs: ToolSpecs, *actions: str) -> None:
        if not isinstance(specs, ToolSpecs):
            raise ValueError(f"GuidedTools requires an initialized `ToolSpecs`, got {type(specs).__name__}.")

        self.specs = {action: specs.tools[action] for action in actions if action in specs.tools}
        self.actions = tuple(self.specs)

    def name_of(self, action: str) -> str:
        return self.specs[action].name

    def action_of(self, name: str) -> str | None:
        return next((action for action in self.actions if self.name_of(action) == name), None)

    def schema_of(self, action: str) -> ToolSchema:
        spec = self.specs[action]
        return ToolSchema(
            type="function",
            function=FunctionDefinition(
                name=spec.name,
                description=spec.description.strip(),
                parameters={
                    "type": "object",
                    "properties": {arg: {"type": a.type, "description": a.description.strip()} for arg, a in spec.arguments.items()},
                    "required": [arg for arg, a in spec.arguments.items() if a.required],
                    "additionalProperties": False,
                },
            ),
        )

    def build(self) -> list[ToolSchema]:
        return [self.schema_of(action) for action in self.actions]

    def parse(self, tool_call: ToolCall, repair: bool = True) -> ToolTurn:
        """
        Resolve the called tool to an action and parse its arguments.

        Args:
            tool_call (ToolCall): The tool call object containing the function name and arguments.
            repair (bool): Whether to attempt to repair malformed JSON arguments.

        Returns:
            ToolTurn: The resolved action and its parsed arguments.

        Raises:
            ValueError: If the tool call type is not "function", if the tool name is unknown, if the arguments cannot be parsed, or if required arguments are missing.
        """

        if tool_call.type != "function":
            raise ValueError(f"Expected a function tool call, got type: {tool_call.type}")

        action = self.action_of(tool_call.function.name)
        if action is None:
            raise ValueError(f"Unknown tool call name: {tool_call.function.name}.")

        spec = self.specs[action]

        try:
            call_args = json.loads(tool_call.function.arguments)

        except json.JSONDecodeError as e:
            if not repair:
                raise ValueError(f"Failed to parse tool call arguments: {e}")

            call_args = json_repair.loads(
                tool_call.function.arguments,
                schema=self.schema_of(action)["function"].get("parameters", {}),
                skip_json_loads=True,
                schema_repair_mode="standard",
            )

        if not isinstance(call_args, dict):
            raise ValueError(f"Parsed tool call arguments are not a dictionary, got {type(call_args)}")

        missing = [arg for arg in spec.arguments if arg not in call_args]
        if missing:
            raise ValueError(f"Missing required argument(s) {missing} for tool call: {tool_call.function.name}.")

        if not all(isinstance(call_args[arg], str) for arg in spec.arguments):
            raise ValueError(f"Argument(s) for tool call: {tool_call.function.name} are not strings.")

        return ToolTurn(
            action=action,
            args={arg: call_args[arg] for arg in spec.arguments},
        )
