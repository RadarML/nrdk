"""Tests for nrdk._cli.export."""

import pytest
import torch
import yaml

from nrdk._cli.export import cli_export


def test_cli_export_directory_mode(tmp_path):
    """Check directory-mode export of weights and config.

    Weights are stripped and `best` is resolved via `checkpoints.yaml`.
    """
    root = tmp_path / "exp"
    (root / ".hydra").mkdir(parents=True)
    with open(root / ".hydra" / "config.yaml", "w") as f:
        yaml.safe_dump({
            "model": {"_target_": "torch.nn.Linear"},
            "transforms": {"foo": "bar"},
            "trainer": {"unused": True},
        }, f)
    (root / "checkpoints.yaml").write_text("best: epoch=1.ckpt\n")
    (root / "checkpoints").mkdir()
    state_dict = {
        "model.layer.weight": torch.ones(2),
        "model._orig_mod.layer2.bias": torch.zeros(2),
    }
    torch.save(
        {"state_dict": state_dict}, root / "checkpoints" / "epoch=1.ckpt")

    cli_export(str(root))

    config = yaml.safe_load((root / "model.yaml").read_text())
    assert config == {
        "model": {"_target_": "torch.nn.Linear"},
        "transforms": {"foo": "bar"},
    }

    exported = torch.load(root / "weights.pth", map_location="cpu")
    assert set(exported.keys()) == {"layer.weight", "layer2.bias"}
    assert torch.equal(
        exported["layer.weight"], state_dict["model.layer.weight"])


def test_cli_export_directory_mode_missing_checkpoints_yaml_raises(tmp_path):
    """A directory with no `checkpoints.yaml` raises `FileNotFoundError`."""
    root = tmp_path / "exp"
    (root / ".hydra").mkdir(parents=True)
    with open(root / ".hydra" / "config.yaml", "w") as f:
        yaml.safe_dump({"model": {}, "transforms": {}}, f)

    with pytest.raises(FileNotFoundError):
        cli_export(str(root))


def test_cli_export_file_mode_uses_given_output_and_skips_config(tmp_path):
    """A direct checkpoint path exports only weights, to the given `output`."""
    ckpt_path = tmp_path / "raw.ckpt"
    torch.save({"state_dict": {"model.w": torch.ones(1)}}, ckpt_path)
    output = tmp_path / "out.pth"

    cli_export(str(ckpt_path), output=str(output))

    exported = torch.load(output, map_location="cpu")
    assert set(exported.keys()) == {"w"}
    assert not (tmp_path / "model.yaml").exists()


def test_cli_export_missing_state_dict_raises(tmp_path):
    """A checkpoint without a `state_dict` key raises `ValueError`."""
    ckpt_path = tmp_path / "raw.ckpt"
    torch.save({"not_state_dict": {}}, ckpt_path)

    with pytest.raises(ValueError):
        cli_export(str(ckpt_path), output=str(tmp_path / "out.pth"))
