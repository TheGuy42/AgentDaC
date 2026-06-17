"""Manual trace construction (`src.custom.convert`).

We build RL traces by hand (no auto-tracer) and feed them through the same consumer chain VERL
uses at train time:

    convert_trajectory  ->  list[Span]  ->  VerlAdapter.adapt  ->  list[Triplet]
                                                                       |
                                                       VerlDaemon.get_train_data_batch (trace_list)

These tests assert the spans carry every field the trainer needs, and that the adapter recovers
valid non-empty triplets + reward for single- and multi-turn trajectories. Pure CPU / in-memory.
"""

from __future__ import annotations

from src.custom import VerlAdapter, convert_trajectory

from tests.unit.factories import make_response, make_rollout, make_trajectory


# --------------------------------------------------------------------------------------------------
# convert_trajectory: span shape
# --------------------------------------------------------------------------------------------------
def test_single_turn_span_shape():
    reward = 1.0
    metrics = {"duration": 1.5, "answer_correct": 1}
    resp = make_response(
        response_id="chatcmpl-aaa",
        content="B",
        prompt_token_ids=[151644, 872, 198],
        response_token_ids=[33, 151645],
    )
    spans = convert_trajectory(make_trajectory([resp], reward=reward, metrics=metrics), make_rollout())

    # one LLM span + one reward span
    assert len(spans) == 2
    llm_span, reward_span = spans

    # LLM span carries exactly the fields the adapter reads
    assert llm_span.name == "openai.chat.completion"
    assert llm_span.attributes["prompt_token_ids"] == [151644, 872, 198]
    assert llm_span.attributes["response_token_ids"] == [33, 151645]
    assert llm_span.attributes["gen_ai.response.id"] == "chatcmpl-aaa"

    # reward span matches the captured auto-trace format
    assert reward_span.name == "agentlightning.annotation"
    assert reward_span.attributes["agentlightning.reward.0.name"] == "primary"
    assert reward_span.attributes["agentlightning.reward.0.value"] == reward
    # custom metrics are flattened under `custom_metrics.*` on the reward span
    assert reward_span.attributes["custom_metrics.duration"] == 1.5
    assert reward_span.attributes["custom_metrics.answer_correct"] == 1


def test_rollout_and_attempt_stamped_on_every_span():
    resp = make_response(response_id="r0", content="x", prompt_token_ids=[1], response_token_ids=[2])
    rollout = make_rollout(rollout_id="ro-xyz", attempt_id="at-xyz")
    spans = convert_trajectory(make_trajectory([resp], reward=0.0, metrics={}), rollout)
    for span in spans:
        assert span.rollout_id == "ro-xyz"
        assert span.attempt_id == "at-xyz"


def test_timestamps_increase_with_reward_last():
    responses = [
        make_response(response_id=f"r{i}", content=str(i), prompt_token_ids=[i], response_token_ids=[i])
        for i in range(3)
    ]
    spans = convert_trajectory(make_trajectory(responses, reward=1.0, metrics={}), make_rollout())
    starts = [s.start_time for s in spans]
    assert starts == sorted(starts)
    assert spans[-1].name == "agentlightning.annotation"
    assert spans[-1].start_time == max(starts)


def test_empty_trajectory_returns_no_spans():
    assert convert_trajectory(make_trajectory([], reward=0.0, metrics={}), make_rollout()) == []


# --------------------------------------------------------------------------------------------------
# VerlAdapter.adapt: spans -> triplets
# --------------------------------------------------------------------------------------------------
def test_single_turn_adapter_triplet():
    reward = 0.75
    resp = make_response(
        response_id="chatcmpl-bbb", content="A", prompt_token_ids=[1, 2, 3, 4], response_token_ids=[42]
    )
    spans = convert_trajectory(make_trajectory([resp], reward=reward, metrics={"duration": 2.0}), make_rollout())

    triplets = VerlAdapter().adapt(spans)

    assert len(triplets) == 1
    t = triplets[0]
    assert t.prompt["token_ids"] == [1, 2, 3, 4]
    assert t.response["token_ids"] == [42]
    assert t.reward == reward
    assert t.metadata["response_id"] == "chatcmpl-bbb"
    assert t.metadata["custom_metrics"] == {"duration": 2.0}


def test_multi_turn_no_silent_drop_and_reward_on_last():
    """Two distinct responses must survive as two triplets; the single terminal reward attaches to
    the last call (FIRST_OCCURRENCE) and is broadcast to all turns later as `final_reward`."""
    reward = 1.0
    responses = [
        make_response(
            response_id=f"chatcmpl-{i}",
            content=str(i),
            prompt_token_ids=[10 + i, 20 + i, 30 + i],
            response_token_ids=[100 + i, 200 + i],
        )
        for i in range(2)
    ]
    spans = convert_trajectory(make_trajectory(responses, reward=reward, metrics={"duration": 3.0}), make_rollout())

    assert len(spans) == 3  # 2 LLM spans + 1 reward span

    triplets = VerlAdapter().adapt(spans)

    # both turns kept (no silent dedup/drop)
    assert [t.metadata["response_id"] for t in triplets] == ["chatcmpl-0", "chatcmpl-1"]
    # reward binds to the last triplet; earlier turns are None until broadcast downstream
    assert triplets[-1].reward == reward
    assert triplets[0].reward is None
    # custom metrics land on the last triplet
    assert triplets[-1].metadata["custom_metrics"] == {"duration": 3.0}


# --------------------------------------------------------------------------------------------------
# Trainer-level: replicate VerlDaemon.get_train_data_batch trace_list construction
# --------------------------------------------------------------------------------------------------
def test_trainer_trace_list_has_non_empty_token_ids():
    """Mirror `AgentModeDaemon.get_train_data_batch` (daemon.py ~837-844): each triplet must yield
    non-empty prompt_ids/response_ids, otherwise the rollout trains on no signal."""
    responses = [
        make_response(response_id=f"r{i}", content=str(i), prompt_token_ids=[1, 2, 3], response_token_ids=[9, 8])
        for i in range(2)
    ]
    triplets = VerlAdapter().adapt(
        convert_trajectory(make_trajectory(responses, reward=1.0, metrics={"duration": 1.0}), make_rollout())
    )

    trace_list = [
        {"prompt_ids": t.prompt.get("token_ids", []), "response_ids": t.response.get("token_ids", [])}
        for t in triplets
    ]

    assert len(trace_list) == 2
    for trace in trace_list:
        assert trace["prompt_ids"], "empty prompt_ids would be silently dropped by the trainer"
        assert trace["response_ids"], "empty response_ids would be silently dropped by the trainer"

    # final reward, as the daemon derives it (last non-None triplet reward)
    final_reward = next((t.reward for t in reversed(triplets) if t.reward is not None), None)
    assert final_reward == 1.0
