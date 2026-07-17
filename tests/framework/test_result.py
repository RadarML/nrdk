"""Tests for nrdk.framework.result."""

import os

import pytest
import yaml
from omegaconf import DictConfig

from nrdk.framework.result import Result


def _write_yaml(path, data) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(data, f)


def _make_result_dir(
    base, name="exp", config=None, checkpoints_index=None, ckpt_files=()
):
    """Build a fake results tree under `base / name`."""
    root = base / name
    (root / ".hydra").mkdir(parents=True, exist_ok=True)
    _write_yaml(
        root / ".hydra" / "config.yaml",
        config if config is not None else {"lr": 0.1, "model": {"name": "m"}})

    if checkpoints_index is not None:
        _write_yaml(root / "checkpoints.yaml", checkpoints_index)

    if ckpt_files:
        (root / "checkpoints").mkdir(parents=True, exist_ok=True)
        for fname in ckpt_files:
            (root / "checkpoints" / fname).write_text("")

    return root


DEFAULT_INDEX = {
    "best": "epoch=1-step=10.ckpt",
    "best_k": {
        "epoch=1-step=10.ckpt": 0.5,
        "epoch=2-step=20.ckpt": 0.6,
    },
}


def test_validate_raises_if_invalid(tmp_path):
    """An invalid result raises the appropriate error."""
    with pytest.raises(ValueError):
        Result.validate(str(tmp_path / "missing"))

    root = tmp_path / "exp"
    root.mkdir()
    with pytest.raises(ValueError):
        Result.validate(str(root))

    root = _make_result_dir(tmp_path, name="exp2", checkpoints_index=None)
    with pytest.raises(ValueError):
        Result.validate(str(root))


def test_validate_passes_for_complete_structure(tmp_path):
    """A complete result directory (hydra config + checkpoints) validates."""
    root = _make_result_dir(
        tmp_path, checkpoints_index=DEFAULT_INDEX,
        ckpt_files=["epoch=1-step=10.ckpt", "epoch=2-step=20.ckpt"])
    # Should not raise.
    Result.validate(str(root))


def test_init_skips_validation_when_disabled(tmp_path):
    """`validate=False` allows constructing a `Result` for an incomplete dir."""
    root = tmp_path / "exp"
    root.mkdir()
    # Structure is incomplete, but validate=False should skip the check.
    result = Result(str(root), validate=False)
    assert result.path == str(root)


def test_find_strict_only_returns_complete_result_dirs(tmp_path):
    """With `strict=True`, only fully-structured result dirs are found."""
    complete = _make_result_dir(
        tmp_path, name="complete", checkpoints_index=DEFAULT_INDEX,
        ckpt_files=["epoch=1-step=10.ckpt"])
    # Only a `.hydra` dir, no `checkpoints.yaml`.
    partial_hydra = tmp_path / "partial_hydra"
    (partial_hydra / ".hydra").mkdir(parents=True)
    _write_yaml(partial_hydra / ".hydra" / "config.yaml", {"a": 1})
    # Only a `checkpoints.yaml`, no `.hydra` dir.
    partial_ckpt = tmp_path / "partial_ckpt"
    partial_ckpt.mkdir()
    _write_yaml(partial_ckpt / "checkpoints.yaml", DEFAULT_INDEX)

    found = Result.find(str(tmp_path), strict=True)

    assert found == [str(complete)]


def test_find_non_strict_returns_any_partial_match(tmp_path):
    """With `strict=False`, partially-structured result dirs are also found."""
    complete = _make_result_dir(
        tmp_path, name="complete", checkpoints_index=DEFAULT_INDEX,
        ckpt_files=["epoch=1-step=10.ckpt"])
    partial_hydra = tmp_path / "partial_hydra"
    (partial_hydra / ".hydra").mkdir(parents=True)
    _write_yaml(partial_hydra / ".hydra" / "config.yaml", {"a": 1})
    partial_ckpt = tmp_path / "partial_ckpt"
    partial_ckpt.mkdir()
    _write_yaml(partial_ckpt / "checkpoints.yaml", DEFAULT_INDEX)

    found = set(Result.find(str(tmp_path), strict=False))

    assert found == {str(complete), str(partial_hydra), str(partial_ckpt)}


def test_find_returns_empty_list_when_nothing_matches(tmp_path):
    """`find` returns an empty list when nothing matches, in either mode."""
    (tmp_path / "unrelated").mkdir()
    assert Result.find(str(tmp_path), strict=True) == []
    assert Result.find(str(tmp_path), strict=False) == []


@pytest.mark.parametrize("omegaconf", [True, False])
def test_config_returns_dictconfig_or_plain_dict(tmp_path, omegaconf):
    """`config(omegaconf=...)` returns a `DictConfig` or a plain `dict`."""
    config = {"lr": 0.1, "model": {"name": "m"}}
    root = _make_result_dir(
        tmp_path, config=config, checkpoints_index=DEFAULT_INDEX,
        ckpt_files=["epoch=1-step=10.ckpt"])
    result = Result(str(root))

    cfg = result.config(omegaconf=omegaconf)
    if omegaconf:
        assert isinstance(cfg, DictConfig)
        assert cfg.lr == 0.1
        assert cfg.model.name == "m"
    else:
        assert type(cfg) is dict
        assert cfg == config


def test_best_returns_joined_checkpoint_path(tmp_path):
    """`best` joins the result root with the index's `best` filename."""
    root = _make_result_dir(
        tmp_path, checkpoints_index=DEFAULT_INDEX,
        ckpt_files=["epoch=1-step=10.ckpt", "epoch=2-step=20.ckpt"])
    result = Result(str(root))

    assert result.best == os.path.join(
        str(root), "checkpoints", "epoch=1-step=10.ckpt")


def test_checkpoints_maps_full_paths_to_scores(tmp_path):
    """`checkpoints` maps full checkpoint paths to their `best_k` scores."""
    root = _make_result_dir(
        tmp_path, checkpoints_index=DEFAULT_INDEX,
        ckpt_files=["epoch=1-step=10.ckpt", "epoch=2-step=20.ckpt"])
    result = Result(str(root))

    checkpoints = result.checkpoints
    assert checkpoints == {
        os.path.join(str(root), "checkpoints", "epoch=1-step=10.ckpt"): 0.5,
        os.path.join(str(root), "checkpoints", "epoch=2-step=20.ckpt"): 0.6,
    }
