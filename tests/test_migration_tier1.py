"""Tier-1 offline checks for the AGL -> verl (main_ppo_sync) migration.

Runnable directly (pytest is not installed in this env):  ``python tests/test_migration_tier1.py``
Covers: config build (incl. in-worker dataset wiring), chat_kwargs temperature/n guards,
metric-schema consistency, the degenerate path, and convert_trajectory round-trips.
No GPU / Ray / vLLM needed.
"""

from __future__ import annotations

import argparse
import pathlib
import types

from omegaconf import OmegaConf

from src.aliases import Response
from src.configs import PromptConfig, RolloutConfig
from src.custom import convert_trajectory
from src.inference import OAIResponse
from src.trainer import RolloutStage
from src.trajectory import Trajectory
from experiments.math_dummy.trainer import MathDummyTrainer
from experiments.math_dummy.run import Runner

CFG_DIR = "experiments/math_dummy/defaults"


def _runner() -> Runner:
    r = Runner()
    r._parser_args = argparse.Namespace(
        project="math_dummy", run="unit", resume=None, config_dir=CFG_DIR,
        seed=123, test_run=False, silent=True, gpus=[0], min_level=1, max_level=5,
    )
    return r


def _trainer() -> MathDummyTrainer:
    """A MathDummyTrainer with just enough state for the unit-level method tests."""
    t = object.__new__(MathDummyTrainer)
    t.rollout_kwargs = RolloutConfig(kwargs={"temperature": 0.7, "top_p": 0.8})
    t.rollout_config = types.SimpleNamespace(temperature=0.7, prompt_length=1024, response_length=4128)
    t.tokenizer = types.SimpleNamespace(encode=lambda s: [1, 2, 3])
    return t


def test_config_build() -> None:
    r = _runner()
    configs = r._load_configs(CFG_DIR)
    cfg = r._build_verl_config(configs, "unit")
    assert "agentlightning" not in cfg
    assert cfg.transfer_queue.enable is True
    # In-worker dataset wiring (no parquet): split markers + custom_cls + embedded args.
    assert cfg.data.train_files == "train" and cfg.data.val_files == "val"
    assert cfg.data.custom_cls.path == "pkg://experiments.math.dataset"
    assert cfg.data.custom_cls.name == "MathDataset"
    assert cfg.data.custom_dataset.min_level == 1 and cfg.data.custom_dataset.max_level == 5
    assert cfg.data.custom_dataset.data_source == "math_dummy"
    assert cfg.data.custom_dataset.seed == 123
    assert cfg.actor_rollout_ref.rollout.mode == "async"
    assert int(cfg.actor_rollout_ref.rollout.nnodes) == 0
    # Agent-loop registration is generated at runtime from trainer_class() (no committed yaml).
    assert cfg.actor_rollout_ref.rollout.agent.default_agent_loop == "MathDummyTrainer"
    alp = cfg.actor_rollout_ref.rollout.agent.agent_loop_config_path
    assert alp.endswith(".yaml") and pathlib.Path(alp).is_file()
    entry = OmegaConf.load(alp)[0]
    assert entry.name == "MathDummyTrainer"
    assert entry._target_ == "experiments.math_dummy.trainer.MathDummyTrainer"
    assert int(cfg.actor_rollout_ref.rollout.response_length) == 4128  # 0*..+(1-0)*4096+32*1
    assert int(cfg.actor_rollout_ref.rollout.max_model_len) == 1024 + 4128
    # full config + agentdac round-trips
    for key in ("actor_rollout_ref", "algorithm", "trainer", "critic", "data", "transfer_queue"):
        assert key in cfg, key
    PromptConfig.model_validate(OmegaConf.to_container(cfg.agentdac.prompt, resolve=True))
    print("PASS test_config_build")


def test_chat_kwargs_guards() -> None:
    t = _trainer()
    # match -> ok
    out = t.chat_kwargs(RolloutStage.TRAIN, {"temperature": 0.7, "top_p": 0.9})
    assert out["temperature"] == 0.7 and out["top_p"] == 0.8  # rollout_kwargs overrides top_p
    # top_p override does NOT raise
    t.chat_kwargs(RolloutStage.TRAIN, {"temperature": 0.7})
    # temperature mismatch on TRAIN -> raise
    t.rollout_kwargs = RolloutConfig(kwargs={}, train_kwargs={"temperature": 0.5})
    try:
        t.chat_kwargs(RolloutStage.TRAIN, {"temperature": 0.7})
        raise AssertionError("expected temperature-mismatch ValueError")
    except ValueError:
        pass
    # 'n' in rollout_kwargs -> raise
    t.rollout_kwargs = RolloutConfig(kwargs={"n": 4})
    try:
        t.chat_kwargs(RolloutStage.VAL, {"temperature": 0.7})
        raise AssertionError("expected 'n' ValueError")
    except ValueError:
        pass
    print("PASS test_chat_kwargs_guards")


def test_reward_extra_info() -> None:
    # main_ppo_sync tolerates per-sample key differences, so reward_extra_info is just the flattened
    # metrics + is_degenerate (no fixed schema, no cross-sample matching required).
    success = MathDummyTrainer._reward_extra_info(
        {"answer_reward": 1.0, "is_correct": True, "parse_success": True}, is_degenerate=False
    )
    assert success == {"answer_reward": 1.0, "is_correct": 1.0, "parse_success": 1.0, "is_degenerate": 0.0}
    degenerate = MathDummyTrainer._reward_extra_info({}, is_degenerate=True)
    assert degenerate == {"is_degenerate": 1.0}
    print("PASS test_reward_extra_info")


def test_stage() -> None:
    t = _trainer()
    assert t._stage({}) == RolloutStage.TRAIN
    assert t._stage({"training_stage": "val"}) == RolloutStage.VAL
    import numpy as np

    assert t._stage({"training_stage": np.str_("train")}) == RolloutStage.TRAIN
    print("PASS test_stage")


def test_degenerate_output() -> None:
    t = _trainer()
    out = t._degenerate_output()
    assert out.reward_score == 0.0 and out.num_turns == 0
    assert set(out.response_mask) == {0}  # non-trainable
    rei = out.extra_fields["reward_extra_info"]
    assert rei == {"is_degenerate": 1.0}
    print("PASS test_degenerate_output")


def _resp(prompt_ids: list[int], response_ids: list[int]) -> OAIResponse:
    comp = Response.model_validate(
        {
            "id": "x", "object": "chat.completion", "created": 0, "model": "m",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"},
                         "finish_reason": "stop", "token_ids": response_ids}],
            "prompt_token_ids": prompt_ids,
        }
    )
    return OAIResponse(comp)


def test_convert_single_and_multi_turn() -> None:
    user = {"role": "user", "content": "q"}

    # single turn
    traj = Trajectory(messages_and_responses=[user, _resp([1, 2, 3], [4, 5])])
    traj.reward = 2.5
    out = convert_trajectory(traj, prompt_length=1024, response_length=1024)
    assert out.prompt_ids == [1, 2, 3]
    assert out.response_ids == [4, 5]
    assert out.response_mask == [1, 1]
    assert out.reward_score == 2.5
    assert out.extra_fields == {}  # reward_extra_info attached by VerlTrainer.run, not convert

    # multi turn: token 6 is an injected (non-generated) token -> masked 0
    traj2 = Trajectory(messages_and_responses=[user, _resp([1, 2, 3], [4, 5]), user, _resp([1, 2, 3, 4, 5, 6], [7, 8])])
    out2 = convert_trajectory(traj2, prompt_length=1024, response_length=1024)
    assert out2.prompt_ids == [1, 2, 3]
    assert out2.response_ids == [4, 5, 6, 7, 8]
    assert out2.response_mask == [1, 1, 0, 1, 1]
    print("PASS test_convert_single_and_multi_turn")


if __name__ == "__main__":
    test_config_build()
    test_chat_kwargs_guards()
    test_reward_extra_info()
    test_stage()
    test_degenerate_output()
    test_convert_single_and_multi_turn()
    print("\nALL TIER-1 TESTS PASSED")
