"""Tests for nrdk.framework.load."""

import pytest
import torch
import yaml

from nrdk.framework.load import load_model

MODEL_YAML = {
    "model": {
        "_target_": "torch.nn.Linear",
        "in_features": 4,
        "out_features": 4,
        "bias": False,
    }
}


def _write_config(path, config=MODEL_YAML) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(config, f)


def _reference_state_dict():
    torch.manual_seed(0)
    return torch.nn.Linear(4, 4, bias=False).state_dict()


def test_load_model_directory_mode(tmp_path):
    """A directory containing `weights.pth`/`model.yaml` loads correctly."""
    state_dict = _reference_state_dict()
    torch.save(state_dict, tmp_path / "weights.pth")
    _write_config(tmp_path / "model.yaml")

    model = load_model(str(tmp_path))

    assert isinstance(model, torch.nn.Linear)
    assert torch.equal(model.weight, state_dict["weight"])


def test_load_model_directory_mode_with_explicit_config(tmp_path):
    """A directory-mode `path` can still take an explicit `config` override."""
    state_dict = _reference_state_dict()
    torch.save(state_dict, tmp_path / "weights.pth")
    other_config = tmp_path / "other.yaml"
    _write_config(other_config)

    model = load_model(str(tmp_path), config=str(other_config))

    assert isinstance(model, torch.nn.Linear)
    assert torch.equal(model.weight, state_dict["weight"])


# File-plus-separate-config-mode loading

def test_load_model_file_mode_with_separate_config(tmp_path):
    """A direct weights file path loads using a separately given config."""
    state_dict = _reference_state_dict()
    weights_path = tmp_path / "ckpt" / "weights.pth"
    weights_path.parent.mkdir(parents=True)
    torch.save(state_dict, weights_path)

    config_path = tmp_path / "config.yaml"
    _write_config(config_path)

    model = load_model(str(weights_path), config=str(config_path))

    assert isinstance(model, torch.nn.Linear)
    assert torch.equal(model.weight, state_dict["weight"])


@pytest.mark.parametrize("freeze", [True, False])
def test_load_model_freeze_flag(tmp_path, freeze):
    """`freeze` controls whether loaded parameters require grad."""
    state_dict = _reference_state_dict()
    torch.save(state_dict, tmp_path / "weights.pth")
    _write_config(tmp_path / "model.yaml")

    model = load_model(str(tmp_path), freeze=freeze, eval=False)

    assert all(not p.requires_grad for p in model.parameters()) == freeze


@pytest.mark.parametrize("eval_flag", [True, False])
def test_load_model_eval_flag(tmp_path, eval_flag):
    """`eval` controls whether the loaded model ends up in training mode."""
    state_dict = _reference_state_dict()
    torch.save(state_dict, tmp_path / "weights.pth")
    _write_config(tmp_path / "model.yaml")

    model = load_model(str(tmp_path), freeze=False, eval=eval_flag)

    assert model.training != eval_flag


def test_load_model_missing_weights_raises(tmp_path):
    """A directory with no `weights.pth` raises `FileNotFoundError`."""
    _write_config(tmp_path / "model.yaml")
    with pytest.raises(FileNotFoundError):
        load_model(str(tmp_path))


def test_load_model_missing_config_raises(tmp_path):
    """A directory with no `model.yaml` raises `FileNotFoundError`."""
    state_dict = _reference_state_dict()
    torch.save(state_dict, tmp_path / "weights.pth")
    # No `model.yaml` written under `tmp_path`.
    with pytest.raises(FileNotFoundError):
        load_model(str(tmp_path))


def test_load_model_file_mode_missing_config_raises(tmp_path):
    """An explicit `config` path that is missing raises `FileNotFoundError`."""
    state_dict = _reference_state_dict()
    weights_path = tmp_path / "weights.pth"
    torch.save(state_dict, weights_path)
    with pytest.raises(FileNotFoundError):
        load_model(str(weights_path), config=str(tmp_path / "missing.yaml"))
