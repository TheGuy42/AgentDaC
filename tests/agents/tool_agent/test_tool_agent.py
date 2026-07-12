"""CPU-only regression tests for the streamlined native-tool-calling agents.

Covers, without a GPU:
  * parsing a delegation turn (native tool parser) and a terminal turn (no tool call);
  * LaTeX / code passing through verbatim (escape-free tag-delimited formats);
  * the implicit-termination agent loop (delegate -> answer directly) via a scripted
    mock inference client.

Reasoning-block splitting is a vLLM reasoning-parser feature (needs a real tokenizer),
so these tests build parsers without one; the whole generation is then the `content`.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from src.inference.client import InferenceClient, InferenceResponse
from src.configs import PromptConfig, DecompConfig
from src.agents.tool_agent import ToolStatelessAgent, ToolPersistentAgent, build_tool_parser
from src.aliases import UserMessage


class _StubTokenizer:
    """Only `get_vocab` is (lazily) touched by the native tool parser; extraction
    operates on the raw string, so an empty vocab is sufficient."""

    def get_vocab(self):
        return {}


def _parser(name="qwen3_xml"):
    return build_tool_parser(name, _StubTokenizer())


def _fp_tc(name, text):
    """A function/parameter (qwen3_xml) tool call."""
    return (
        f"<tool_call>\n<function={name}>\n"
        f"<parameter=text>\n{text}\n</parameter>\n"
        f"</function>\n</tool_call>"
    )


# --------------------------------------------------------------- parser tests

def test_delegation_turn_parse_latex_verbatim():
    p = _parser()
    turn = p.parse(_fp_tc("create_new_sub_agent", "Evaluate \\boxed{d8d7} and \\frac{1}{2} for a < b"))
    assert turn.has_tool_call
    call = turn.tool_call
    assert call.function.name == "create_new_sub_agent"
    assert json.loads(call.function.arguments)["text"] == "Evaluate \\boxed{d8d7} and \\frac{1}{2} for a < b"


def test_terminal_turn_is_the_answer():
    # No tool call -> terminal; the content is the answer (verbatim, no escaping).
    p = _parser()
    answer = "The best move is \\boxed{d8d7}."
    turn = p.parse(answer)
    assert not turn.has_tool_call
    assert turn.content == answer


def test_unsupported_json_format_rejected():
    with pytest.raises(ValueError):
        build_tool_parser("hermes", _StubTokenizer())


def test_stop_tag_mapping():
    from src.agents.tool_agent.parsing import STOP_TAGS

    assert STOP_TAGS["seed_oss"] == "</seed:tool_call>"
    assert STOP_TAGS["qwen3_xml"] == "</tool_call>"


# ----------------------------------------------------------- agent-loop tests

class _MockResponse(InferenceResponse):
    def __init__(self, content):
        self._c = content

    @property
    def content(self):
        return self._c

    @property
    def tool_calls(self):
        return None

    @property
    def finish_reason(self):
        return "stop"

    @property
    def total_tokens(self):
        return max(1, len(self._c.split()))


class _MockClient(InferenceClient):
    """Delegates once (fresh sub-agent) when permitted on the first turn, then answers
    directly (no tool call). A leaf, which cannot delegate, answers immediately."""

    def update_kwargs(self, kwargs, **_):
        return kwargs

    async def chat(self, messages, **kwargs):
        n_assistant = sum(1 for m in messages if m.get("role") == "assistant")
        status = [m for m in messages if str(m.get("content", "")).startswith("[status]")]
        can_delegate = bool(status) and "create_new_sub_agent" in str(status[-1]["content"])
        if can_delegate and n_assistant == 0:
            return _MockResponse(_fp_tc("create_new_sub_agent", "solve the sub-task"))
        return _MockResponse("e2e4")  # terminal: plain-text final answer, no tool call


_PROMPTS = PromptConfig(mode="text", system_root="ROOT", system_inter="INTER", system_leaf="LEAF")


def _run(agent_cls, **decomp):
    agent = agent_cls(
        client=_MockClient(),
        prompt_config=_PROMPTS,
        decomp_config=DecompConfig(**decomp),
        tool_parser=_parser(),
    )
    traj = asyncio.run(agent.chat(UserMessage(role="user", content="Solve.")))
    return agent, traj


def test_stateless_delegates_then_answers():
    agent, traj = _run(ToolStatelessAgent, max_depth=1, max_rounds=3, max_tasks=2)
    roles = [m["role"] for m in traj.messages()]
    assert roles.count("assistant") == 2  # delegated, then answered directly
    assert "tool" in roles  # sub-agent answer returned as a tool message
    assert agent.parse_answer(traj.messages()[-1]) == "e2e4"
    assert agent.metrics["direct_calls"] == 2 and agent.metrics["direct_tasks"] == 1
    assert agent.metrics["direct_agents"] == 1 and agent.metrics["direct_responses_completed"] == 1


def test_persistent_delegates_then_answers():
    agent, traj = _run(ToolPersistentAgent, max_depth=1, max_rounds=3, max_tasks=2)
    assert any(m["role"] == "tool" for m in traj.messages())
    assert agent.parse_answer(traj.messages()[-1]) == "e2e4"
    assert agent.metrics["direct_agents"] == 1


def test_leaf_answers_immediately():
    agent, traj = _run(ToolStatelessAgent, max_depth=0, max_rounds=3, max_tasks=2)
    assert [m["role"] for m in traj.messages()].count("assistant") == 1
    assert agent.parse_answer(traj.messages()[-1]) == "e2e4"


def test_leaf_advertises_no_tools():
    agent = ToolStatelessAgent(
        client=_MockClient(),
        prompt_config=PromptConfig(mode="text", system_root="LEAF"),
        decomp_config=DecompConfig(max_depth=0, max_rounds=3, max_tasks=2),
        tool_parser=_parser(),
    )
    assert agent.tool_map == {}
