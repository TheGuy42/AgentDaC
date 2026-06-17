"""Per-stage rollout kwargs (`src.configs.RolloutConfig.get_kwargs`)."""

from __future__ import annotations

from src.configs import RolloutConfig


def test_base_kwargs_used_for_all_stages():
    cfg = RolloutConfig(kwargs={"temperature": 0.7})
    for stage in ("train", "val", "test"):
        assert cfg.get_kwargs(stage) == {"temperature": 0.7}


def test_stage_kwargs_override_base():
    cfg = RolloutConfig(
        kwargs={"temperature": 0.7, "max_completion_tokens": 256},
        train_kwargs={"temperature": 1.0},
        val_kwargs={"temperature": 0.0},
    )
    assert cfg.get_kwargs("train") == {"temperature": 1.0, "max_completion_tokens": 256}
    assert cfg.get_kwargs("val") == {"temperature": 0.0, "max_completion_tokens": 256}
    # test stage has no overrides -> base only
    assert cfg.get_kwargs("test") == {"temperature": 0.7, "max_completion_tokens": 256}


def test_unknown_stage_falls_back_to_base():
    cfg = RolloutConfig(kwargs={"temperature": 0.7}, train_kwargs={"temperature": 1.0})
    assert cfg.get_kwargs("bogus") == {"temperature": 0.7}


def test_get_kwargs_does_not_mutate_config():
    cfg = RolloutConfig(kwargs={"temperature": 0.7}, train_kwargs={"top_p": 0.9})
    result = cfg.get_kwargs("train")
    result["temperature"] = 999
    # mutating the returned dict must not leak back into the config
    assert cfg.kwargs == {"temperature": 0.7}
    assert cfg.train_kwargs == {"top_p": 0.9}
