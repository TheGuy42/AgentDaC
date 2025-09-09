from __future__ import annotations
from typing import Any
import json_repair

from src.agents.strategies.base import ParseStrategy, AgentTurn, TurnAction
from src.agents.exceptions import ParseError, InvalidActionError
from src.configs import DecompConfig
from src.utils.logging import create_logger

logger = create_logger(__name__)


class JsonParseStrategy(ParseStrategy):
    """JSON-based parsing strategy using OpenAI's structured output."""
    
    def get_allowed_actions(self, decomp_config: DecompConfig, current_depth: int) -> list[TurnAction]:
        """Determine allowed actions based on decomposition rules."""
        is_leaf = current_depth >= decomp_config.max_depth
        has_rounds = decomp_config.total_rounds < decomp_config.max_rounds
        tasks_available = decomp_config.total_tasks < decomp_config.max_tasks

        allowed = [TurnAction.ANSWER]
        if has_rounds:
            allowed.append(TurnAction.THINK)
            if (not is_leaf) and tasks_available:
                allowed.append(TurnAction.ISSUE_TASK)

        return allowed
    
    def prepare_call_kwargs(self, allowed_actions: list[TurnAction], **kwargs) -> dict[str, Any]:
        """Prepare JSON schema for structured output."""
        # Build JSON schema
        json_schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "action": {"type": "string", "enum": [a.value for a in allowed_actions]},
                "text": {"type": "string"},
            },
            "required": ["action", "text"],
        }

        schema_descriptor = {
            "name": "assistant_turn",
            "description": "Json schema for a single assistant turn.",
            "strict": True,
            "schema": json_schema,
        }
        
        # Update kwargs for JSON schema response
        extra_body: dict = kwargs.setdefault("extra_body", {})
        extra_body.setdefault("include_stop_str_in_output", True)
        kwargs["response_format"] = {"type": "json_schema", "json_schema": schema_descriptor}
        
        return kwargs
    
    def parse_response(self, content: str, allowed_actions: list[TurnAction]) -> AgentTurn:
        """Parse JSON response into AgentTurn."""
        if not isinstance(content, str):
            raise ParseError("Content to parse must be a string", content, "JSON")

        try:
            json_obj = json_repair.loads(content, skip_json_loads=True)
        except Exception as e:
            logger.debug(f"Failed to repair/parse JSON: {e}")
            logger.debug(f"Raw content was: {content}")
            raise ParseError(f"Content is not valid JSON: {e}", content, "JSON")
            
        if not isinstance(json_obj, dict):
            logger.debug(f"Parsed JSON object: {json_obj}")
            raise ParseError(f"Parsed content is not a dictionary, got {type(json_obj)}", content, "JSON")

        # Validate required fields
        if "action" not in json_obj:
            raise ParseError("Missing required field 'action'", content, "JSON")
        if "text" not in json_obj:
            raise ParseError("Missing required field 'text'", content, "JSON")
            
        action_val = json_obj["action"]
        if not isinstance(action_val, str):
            raise ParseError(f"Field 'action' must be a string, got {type(action_val)}", content, "JSON")

        text_val = json_obj["text"]
        if not isinstance(text_val, str):
            raise ParseError(f"Field 'text' must be a string, got {type(text_val)}", content, "JSON")

        try:
            action = TurnAction(action_val)
        except ValueError:
            raise ParseError(f"Invalid action '{action_val}'. Must be one of: {[a.value for a in TurnAction]}", content, "JSON")
        
        if action not in allowed_actions:
            raise InvalidActionError(action.value, [a.value for a in allowed_actions], content=content, strategy="JSON")

        return AgentTurn(action=action, text=text_val.strip(), raw=json_obj)
    
    def parse_final_answer(self, content: str) -> str:
        """Parse final answer from JSON response."""
        try:
            turn = self.parse_response(content, [TurnAction.ANSWER])
            return turn.text
        except Exception as e:
            logger.error(f"Failed to parse final answer: {e}")
            return content