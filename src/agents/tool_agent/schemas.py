from openai.types.shared_params.function_definition import FunctionDefinition
from src.aliases import ToolSchema

def tool_schema(name: str, desc: str, arg_name: str, arg_desc: str) -> ToolSchema:
    """A function tool taking a single required string argument."""
    return ToolSchema(
        type="function",
        function=FunctionDefinition(
            name=name,
            description=desc,
            parameters={
                "type": "object",
                "properties": {arg_name: {"type": "string", "description": arg_desc}},
                "required": [arg_name],
                "additionalProperties": False,
            },
        ),
    )
