"""Round-trip tests for ``src.utils.convert.trajectory_to_spans``.

Validates that a finished ``Trajectory`` (holding raw ``ChatCompletion`` objects) converts
into AgentLightning spans that the stock ``TracerTraceToTriplet`` adapter turns into correct
training triplets — mirroring what VERL's daemon does at training time.

Runnable with pytest *or* directly: ``python tests/test_convert.py``.
"""

from __future__ import annotations

import pathlib
import sys
from datetime import datetime

MODULE_DIR = pathlib.Path(__file__).parent.parent.resolve()
if str(MODULE_DIR) not in sys.path:
    sys.path.append(str(MODULE_DIR))

from agentlightning import Span, TracerTraceToTriplet, get_reward_value  # noqa: E402
from openai.types.chat.chat_completion import ChatCompletion, Choice  # noqa: E402
from openai.types.chat.chat_completion_message import ChatCompletionMessage  # noqa: E402

from src.trajectory import History, Trajectory  # noqa: E402
from src.utils.convert import LLM_CALL_SPAN_NAME, convert_trajectory  # noqa: E402


def _fake_completion(rid: str, prompt_ids: list[int], response_ids: list[int], text: str) -> ChatCompletion:
    """Build a ChatCompletion with vLLM-style token ids attached as extra fields."""
    message = ChatCompletionMessage(role="assistant", content=text)
    choice = Choice(index=0, finish_reason="stop", message=message)
    resp = ChatCompletion(id=rid, choices=[choice], created=0, model="m", object="chat.completion")
    resp.__dict__["prompt_token_ids"] = prompt_ids
    choice.__dict__["token_ids"] = response_ids
    return resp


def _sample_trajectory() -> Trajectory:
    r1 = _fake_completion("resp-1", [1, 2, 3], [4, 5], "turn1")
    r2 = _fake_completion("resp-2", [1, 2, 3, 4, 5, 6], [7, 8, 9], "turn2")
    sub = _fake_completion("resp-sub", [10, 11], [12], "subturn")
    return Trajectory(
        messages_and_responses=[{"role": "user", "content": "hi"}, r1, r2],
        additional_histories=[History(messages_and_responses=[sub])],
        start_time=datetime.now(),
    )


def test_trajectory_to_spans_shape() -> None:
    spans = convert_trajectory(_sample_trajectory(), reward=1.0)

    llm_spans = [s for s in spans if s.name == LLM_CALL_SPAN_NAME]
    reward_spans = [s for s in spans if get_reward_value(s) is not None]

    assert len(llm_spans) == 3, "expected one span per ChatCompletion (incl. sub-agent history)"
    assert len(reward_spans) == 1, "expected exactly one reward span"

    for s in llm_spans:
        assert s.attributes["prompt_token_ids"], "prompt token ids must be non-empty"
        assert s.attributes["response_token_ids"], "response token ids must be non-empty"

    times = [s.start_time for s in spans]
    assert times == sorted(times), "span timestamps must be strictly increasing / ordered"
    assert reward_spans[0].start_time == max(times), "reward span must come last"


def test_adapter_round_trip() -> None:
    spans_core = convert_trajectory(_sample_trajectory(), reward=1.0)
    # Mimic the runner: SpanCoreFields -> Span with rollout/attempt/sequence ids.
    spans = [Span.from_core_fields(c, rollout_id="r0", attempt_id="a0", sequence_id=i) for i, c in enumerate(spans_core)]

    triplets = TracerTraceToTriplet().adapt(spans)

    assert len(triplets) == 3, "one triplet per LLM call"
    for t in triplets:
        assert t.prompt.get("token_ids"), "triplet prompt token ids must be non-empty"
        assert t.response.get("token_ids"), "triplet response token ids must be non-empty"
    # FIRST_OCCURRENCE matching attaches the scalar reward to the final LLM call.
    assert triplets[-1].reward == 1.0, "final triplet should carry the trajectory reward"
    assert all(t.reward in (None, 1.0) for t in triplets)


def test_missing_token_ids_are_empty() -> None:
    # No token-id extras set -> empty lists (VERL would drop these, but conversion must not crash).
    bare = _fake_completion("bare", [], [], "x")
    bare.__dict__.pop("prompt_token_ids", None)
    bare.choices[0].__dict__.pop("token_ids", None)
    traj = Trajectory(messages_and_responses=[bare], start_time=datetime.now())

    spans = convert_trajectory(traj, reward=0.0)
    llm_span = next(s for s in spans if s.name == LLM_CALL_SPAN_NAME)
    assert llm_span.attributes["prompt_token_ids"] == []
    assert llm_span.attributes["response_token_ids"] == []


def _run_all() -> None:
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("All convert tests passed.")


if __name__ == "__main__":
    _run_all()
