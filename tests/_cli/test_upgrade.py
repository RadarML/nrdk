"""Tests for nrdk._cli.upgrade."""

import pytest

from nrdk._cli.upgrade import cli_upgrade


def _make_result(tmp_path, name, config_text):
    """Build a minimal results tree with the given raw `config.yaml` text.

    Includes `.hydra/config.yaml` and a stub `checkpoints.yaml`.
    """
    root = tmp_path / name
    (root / ".hydra").mkdir(parents=True)
    (root / ".hydra" / "config.yaml").write_text(config_text)
    (root / "checkpoints.yaml").write_text("best: foo.ckpt\n")
    return root


def test_cli_upgrade_dry_run_reports_without_modifying(tmp_path, capsys):
    """`dry_run=True` reports matches but leaves the config untouched."""
    config_text = "model:\n  _target_: old.module.Class\n  param: 1\n"
    root = _make_result(tmp_path, "exp", config_text)

    cli_upgrade("old.module.Class", path=str(tmp_path), dry_run=True)

    out = capsys.readouterr().out
    assert "Found 1 occurrence(s) of 'old.module.Class'" in out
    assert (root / ".hydra" / "config.yaml").read_text() == config_text


def test_cli_upgrade_rewrites_matching_target(tmp_path, capsys):
    """Without `dry_run`, matching `_target_` entries are replaced in place."""
    root = _make_result(
        tmp_path, "exp", "model:\n  _target_: old.module.Class\n  param: 1\n")

    cli_upgrade("old.module.Class", to="new.module.Class", path=str(tmp_path))

    new_text = (root / ".hydra" / "config.yaml").read_text()
    assert "_target_: new.module.Class" in new_text
    assert "old.module.Class" not in new_text
    assert "Upgrading 1 occurrence(s)" in capsys.readouterr().out


def test_cli_upgrade_only_modifies_matching_results(tmp_path):
    """Only results whose config actually references `target` are rewritten."""
    matching = _make_result(
        tmp_path, "matching", "model:\n  _target_: old.module.Class\n")
    other_text = "model:\n  _target_: other.module.Class\n"
    other = _make_result(tmp_path, "other", other_text)

    cli_upgrade("old.module.Class", to="new.module.Class", path=str(tmp_path))

    assert "new.module.Class" in (
        matching / ".hydra" / "config.yaml").read_text()
    assert (other / ".hydra" / "config.yaml").read_text() == other_text


def test_cli_upgrade_requires_to_unless_dry_run(tmp_path):
    """Omitting `to` without `dry_run` raises `ValueError`."""
    _make_result(tmp_path, "exp", "model:\n  _target_: old.module.Class\n")

    with pytest.raises(ValueError):
        cli_upgrade("old.module.Class", path=str(tmp_path), dry_run=False)
