from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from experiments.chess.chess_engine.config import ChessConfig, EngineConfig
from src.backends.vllm.config import InferenceConfig, VllmConfig
from src.configs import DataConfig, DecompConfig, PromptConfig, RolloutConfig


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO_ROOT / "experiments"


def config_files(name: str) -> list[Path]:
    return sorted(CONFIG_ROOT.glob(f"*/configs/*/{name}"))


def test_structural_configs_reject_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        PromptConfig.model_validate({"mode": "text", "unknown_prompt_option": True})

    with pytest.raises(ValidationError, match="extra_forbidden"):
        VllmConfig.model_validate(
            {
                "server": {
                    "model_name": "model",
                    "unknown_server_option": True,
                }
            }
        )


def test_extension_configs_preserve_unknown_fields() -> None:
    data = DataConfig.model_validate({"train_size": 0, "dataset_variant": "custom"})
    assert data.dataset_variant == "custom"  # type: ignore[attr-defined]
    assert data.model_dump()["dataset_variant"] == "custom"


def test_art_model_config_preserves_backend_extensions() -> None:
    pytest.importorskip("art")
    from src.backends.art.config import ModelConfig

    model = ModelConfig.model_validate(
        {
            "base_model": "model",
            "backend_extension": {"enabled": True},
        }
    )
    assert model.model_dump()["backend_extension"] == {"enabled": True}


@pytest.mark.parametrize("field", ["max_depth", "max_tasks", "max_rounds"])
def test_decomposition_budgets_allow_zero_and_reject_negative(field: str) -> None:
    assert getattr(DecompConfig(**{field: 0}), field) == 0
    with pytest.raises(ValidationError):
        DecompConfig(**{field: -1})


@pytest.mark.parametrize("field", ["train_size", "val_size", "test_size"])
def test_dataset_sizes_allow_none_or_zero_and_reject_negative(field: str) -> None:
    assert getattr(DataConfig(**{field: None}), field) is None
    assert getattr(DataConfig(**{field: 0}), field) == 0
    with pytest.raises(ValidationError):
        DataConfig(**{field: -1})


@pytest.mark.parametrize("field", ["group_size", "max_concurrency"])
def test_inference_limits_must_be_positive(field: str) -> None:
    with pytest.raises(ValidationError):
        InferenceConfig(**{field: 0})


@pytest.mark.parametrize("field", ["threads", "hash_mb"])
def test_engine_resources_must_be_positive(field: str) -> None:
    with pytest.raises(ValidationError):
        EngineConfig(**{field: 0})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("mate_score", 0),
        ("win_scale", 0),
        ("mate_margin", 0),
        ("mate_margin", 0.5),
        ("mate_decay", -0.001),
    ],
)
def test_chess_reward_parameters_are_bounded(field: str, value: float) -> None:
    with pytest.raises(ValidationError):
        ChessConfig(limit={"depth": 1}, **{field: value})


def test_art_training_limits_are_validated() -> None:
    pytest.importorskip("art")
    from src.backends.art.config import ArtTrainConfig

    for field in ("epochs", "num_groups", "group_size", "val_log_steps"):
        with pytest.raises(ValidationError):
            ArtTrainConfig(**{field: 0})

    with pytest.raises(ValidationError):
        ArtTrainConfig(max_exceptions=-1)


@pytest.mark.parametrize(
    ("file_name", "config_type"),
    [
        ("data_config.yaml", DataConfig),
        ("prompt_config.yaml", PromptConfig),
        ("decomp_config.yaml", DecompConfig),
        ("verl_rollout_config.yaml", RolloutConfig),
        ("art_rollout_config.yaml", RolloutConfig),
        ("vllm_rollout_config.yaml", RolloutConfig),
        ("vllm_config.yaml", VllmConfig),
        ("engine_config.yaml", EngineConfig),
        ("chess_config.yaml", ChessConfig),
    ],
)
def test_checked_in_typed_configs_load(file_name: str, config_type: type) -> None:
    paths = config_files(file_name)
    assert paths, f"No checked-in {file_name} files found"
    for path in paths:
        config_type.load_from_path(path, do_raise=True)


def test_checked_in_art_configs_load() -> None:
    pytest.importorskip("art")
    from src.backends.art.config import ArtConfig

    paths = config_files("art_config.yaml")
    assert paths, "No checked-in art_config.yaml files found"
    for path in paths:
        ArtConfig.load_from_path(path, do_raise=True)


def walk_items(value: Any):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key, child
            yield from walk_items(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_items(child)


def test_checked_in_repository_asset_paths_are_relative_and_exist() -> None:
    path_keys = {
        "system_root",
        "system_inter",
        "system_leaf",
        "chat_template",
        "custom_chat_template",
    }

    for path in sorted(CONFIG_ROOT.glob("*/configs/*/*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        for key, value in walk_items(data):
            if key not in path_keys or value is None or not isinstance(value, str):
                continue
            asset_path = Path(value)
            assert not asset_path.is_absolute(), f"{path}: {key} must be repository-relative"
            assert (REPO_ROOT / asset_path).is_file(), f"{path}: missing {value}"


def test_configs_contain_no_machine_specific_home_paths() -> None:
    for path in sorted(CONFIG_ROOT.glob("*/configs/*/*.yaml")):
        text = path.read_text(encoding="utf-8")
        assert "/home/" not in text, path
        assert "/Users/" not in text, path
