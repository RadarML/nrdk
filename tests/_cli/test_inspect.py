"""Tests for nrdk._cli.inspect."""

import os

import pytest
import torch

from nrdk._cli.inspect import cli_inspect


def test_cli_inspect_file_directly(tmp_path, capsys):
    """Check that a direct state-dict file is printed as a parameter tree.

    Tree keys should be derived from the dotted parameter names.
    """
    state_dict = {
        "encoder.layer1.weight": torch.randn(4, 4),
        "encoder.layer1.bias": torch.randn(4),
        "decoder.weight": torch.randn(2, 2),
    }
    path = tmp_path / "model.pt"
    torch.save(state_dict, path)

    cli_inspect(str(path))

    out = capsys.readouterr().out
    assert "encoder" in out
    assert "decoder" in out


def test_cli_inspect_unwraps_lightning_state_dict(tmp_path, capsys):
    """Check that a pytorch-lightning-style checkpoint is unwrapped.

    A `state_dict` key alongside other top-level keys (e.g. `epoch`) should
    be unwrapped before inspection.
    """
    checkpoint = {
        "state_dict": {"model.layer.weight": torch.randn(3, 3)},
        "epoch": 3,
    }
    path = tmp_path / "ckpt.ckpt"
    torch.save(checkpoint, path)

    cli_inspect(str(path))

    assert "model" in capsys.readouterr().out


def test_cli_inspect_directory_mode_selects_most_recent_checkpoint(
    tmp_path, capsys
):
    """Check that directory mode selects the most recently modified checkpoint.

    Any `*.ckpt/pt/pth` file under the directory is a candidate.
    """
    old_path = tmp_path / "old.ckpt"
    torch.save({
        "old_a.weight": torch.randn(2, 2),
        "old_b.weight": torch.randn(2, 2),
    }, old_path)
    new_path = tmp_path / "new.pt"
    torch.save({
        "new_a.weight": torch.randn(2, 2),
        "new_b.weight": torch.randn(2, 2),
    }, new_path)

    now = os.path.getmtime(new_path)
    os.utime(old_path, (now - 100, now - 100))
    os.utime(new_path, (now + 100, now + 100))

    cli_inspect(str(tmp_path))

    out = capsys.readouterr().out
    assert str(new_path) in out
    assert "new_a" in out and "new_b" in out
    assert "old_a" not in out and "old_b" not in out


def test_cli_inspect_raises_when_directory_has_no_checkpoint(tmp_path):
    """A directory with no `*.ckpt/pt/pth` file raises `FileNotFoundError`."""
    with pytest.raises(FileNotFoundError):
        cli_inspect(str(tmp_path))


def test_cli_inspect_single_parameter_checkpoint(tmp_path, capsys):
    """A checkpoint with exactly one tensor doesn't crash.

    `_collapse_singletons` can fold a fully-unbranching tree all the way
    down to a bare `Param` rather than a dict; `cli_inspect` must not assume
    the collapsed root is always a dict.
    """
    path = tmp_path / "single.pt"
    torch.save({"only.weight": torch.randn(2, 2)}, path)

    cli_inspect(str(path))

    assert "only.weight" in capsys.readouterr().out
