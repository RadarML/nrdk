"""Tests for nrdk._cli.validate."""

import os

from nrdk._cli.validate import cli_validate


def _make_result(tmp_path, name, complete=True):
    """Build a fake results tree under `tmp_path / name`.

    If `complete`, all files `cli_validate` checks for are present;
    otherwise only `.hydra/config.yaml` is written.
    """
    root = tmp_path / name
    (root / ".hydra").mkdir(parents=True)
    (root / ".hydra" / "config.yaml").write_text("model: {}\n")

    if complete:
        (root / "checkpoints").mkdir()
        (root / "checkpoints" / "last.ckpt").write_text("")
        (root / "eval").mkdir()
        (root / "checkpoints.yaml").write_text("best: last.ckpt\n")
        (root / "events.out.tfevents.123").write_text("")

    return root


def test_cli_validate_reports_all_complete(tmp_path, capsys):
    """A fully-structured result directory is reported as complete."""
    _make_result(tmp_path, "exp", complete=True)

    cli_validate(str(tmp_path))

    out = capsys.readouterr().out
    assert "All 1 results directories are complete." in out


def test_cli_validate_reports_missing_files(tmp_path, capsys):
    """An incomplete result directory is counted and listed by default."""
    _make_result(tmp_path, "full_result", complete=True)
    partial = _make_result(tmp_path, "partial_result", complete=False)

    cli_validate(str(tmp_path))

    out = capsys.readouterr().out
    assert "Found 2 results directories with 1 incomplete results." in out
    assert os.path.relpath(str(partial), str(tmp_path)) in out


def test_cli_validate_show_all_includes_complete_rows(tmp_path, capsys):
    """`show_all=True` also lists directories with no missing files."""
    full = _make_result(tmp_path, "full_result", complete=True)
    _make_result(tmp_path, "partial_result", complete=False)

    cli_validate(str(tmp_path), show_all=False)
    out_default = capsys.readouterr().out
    cli_validate(str(tmp_path), show_all=True)
    out_all = capsys.readouterr().out

    full_relpath = os.path.relpath(str(full), str(tmp_path))
    assert full_relpath not in out_default
    assert full_relpath in out_all
