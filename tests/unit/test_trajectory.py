"""Trajectory message conversion (`src.trajectory`)."""

from __future__ import annotations

from src.aliases import UserMessage
from src.trajectory import Trajectory, get_messages

from tests.unit.factories import make_response


def test_response_becomes_assistant_message():
    resp = make_response(response_id="r0", content="hello", prompt_token_ids=[1], response_token_ids=[2])
    messages = get_messages([UserMessage(role="user", content="hi"), resp])

    assert messages[0] == {"role": "user", "content": "hi"}
    assert messages[1]["role"] == "assistant"
    assert messages[1]["content"] == "hello"
    assert "tool_calls" not in messages[1]  # omitted when there are none


def test_none_content_is_coerced_to_empty_string():
    # tokenizer chat templates require string content
    messages = get_messages([{"role": "assistant", "content": None}])
    assert messages[0]["content"] == ""


def test_for_logging_marks_trainable_per_message():
    resp = make_response(response_id="r0", content="A", prompt_token_ids=[1], response_token_ids=[2])
    traj = Trajectory(
        messages_and_responses=[UserMessage(role="user", content="q"), resp],
        reward=0.5,
        metrics={"duration": 1.0},
    )
    logged = traj.for_logging()

    assert logged["reward"] == 0.5
    assert logged["metrics"] == {"duration": 1.0}
    # user message is not trainable, the model response is
    assert logged["messages"][0]["trainable"] is False
    assert logged["messages"][1]["trainable"] is True


def test_messages_helper_matches_get_messages():
    resp = make_response(response_id="r0", content="A", prompt_token_ids=[1], response_token_ids=[2])
    items = [UserMessage(role="user", content="q"), resp]
    traj = Trajectory(messages_and_responses=items)
    assert traj.messages() == get_messages(items)
