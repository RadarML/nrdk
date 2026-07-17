"""Tests for `nrdk.tss._cli`.

`_cli_main` is just `tyro.cli(_cli)`, so we test the underlying `_cli(...)`
function directly with plain keyword arguments, bypassing `tyro`'s
argv-parsing entirely. Testing `tyro`'s own CLI-parsing behavior is out of
scope; `_cli_main` itself is not tested beyond this.
"""

from io import StringIO

import numpy as np
import pandas as pd
import pytest
import yaml

from nrdk.tss._cli import _cli

# `_cli`'s default pattern only matches flat `<experiment>.npz` files, not
# the `<exp>/eval/seq.npz` layout `_make_results_tree` writes below, so all
# fixtures need an explicit pattern with both `experiment` and `trace` groups.
_PATTERN = r"^(?P<experiment>(.*))/eval/(?P<trace>(.*))\.npz$"


def _make_results_tree(tmp_path, experiments):
    """Write `<exp>/eval/seq.npz` result files for each experiment.

    Args:
        tmp_path: root directory (typically pytest's `tmp_path` fixture).
        experiments: mapping `{experiment_name: {array_name: array, ...}}`.
    """
    for exp, arrays in experiments.items():
        p = tmp_path / exp / "eval" / "seq.npz"
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(p, **arrays)


def test_cli_success_returns_zero_and_prints_valid_csv(tmp_path, capsys):
    """A successful run returns 0 and prints a CSV parseable by pandas."""
    rng = np.random.default_rng(0)
    _make_results_tree(tmp_path, {
        "expA": {"loss": rng.normal(size=20) + 1.0},
        "expB": {"loss": rng.normal(size=20) + 2.0},
    })

    rc = _cli(str(tmp_path), pattern=_PATTERN, key="loss")

    assert rc == 0
    out = capsys.readouterr().out
    df = pd.read_csv(StringIO(out), index_col="name")
    assert sorted(df.index) == ["expA", "expB"]
    assert "abs/mean" in df.columns


def test_cli_no_match_returns_negative_one_with_hint_message(tmp_path, capsys):
    """When no files match, `_cli` returns -1 with a hint message."""
    _make_results_tree(tmp_path, {"expA": {"loss": np.arange(10, dtype=float)}})

    rc = _cli(str(tmp_path), pattern=r"^nomatch$")

    assert rc == -1
    out = capsys.readouterr().out
    assert "No result files found!" in out
    assert "follow_symlinks" in out


def test_cli_config_file_supplies_defaults_unless_overridden(tmp_path, capsys):
    """Unset values come from `--config`; an explicit argument still wins."""
    rng = np.random.default_rng(1)
    _make_results_tree(tmp_path, {
        "expA": {
            "loss": rng.normal(size=20) + 1.0,
            "other_metric": rng.normal(size=20) + 100.0,
        },
    })
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump({"key": "other_metric", "pattern": _PATTERN}, f)

    rc = _cli(str(tmp_path), config=str(config_path))

    assert rc == 0
    df = pd.read_csv(StringIO(capsys.readouterr().out), index_col="name")
    # `other_metric` is centered around 100, `loss` around 1 -- confirms the
    # config-supplied `key` (not the CLI's hardcoded default "loss") was used.
    assert df.loc["expA", "abs/mean"] > 50  # type: ignore

    # Explicitly passing key="loss" should win over the config's
    # key="other_metric".
    rc2 = _cli(str(tmp_path), config=str(config_path), key="loss")

    assert rc2 == 0
    df2 = pd.read_csv(StringIO(capsys.readouterr().out), index_col="name")
    assert df2.loc["expA", "abs/mean"] < 50  # type: ignore


def test_cli_baseline_adds_relative_and_pct_columns(tmp_path, capsys):
    """Passing `baseline` adds `rel/*` and `pct/*` columns to the CSV output."""
    rng = np.random.default_rng(3)
    _make_results_tree(tmp_path, {
        "base": {"loss": rng.normal(size=30) + 1.0},
        "treatment": {"loss": rng.normal(size=30) + 5.0},
    })

    rc = _cli(str(tmp_path), pattern=_PATTERN, key="loss", baseline="base")

    assert rc == 0
    df = pd.read_csv(StringIO(capsys.readouterr().out), index_col="name")
    assert "rel/mean" in df.columns
    assert "pct/mean" in df.columns
    # Baseline relative to itself is 0; treatment is offset by roughly +4.
    assert df.loc["base", "rel/mean"] == pytest.approx(0.0)
    assert df.loc["treatment", "rel/mean"] > 0  # type: ignore


def test_cli_experiments_filter_restricts_output_rows(tmp_path, capsys):
    """The explicit `experiments` list restricts which rows are computed."""
    rng = np.random.default_rng(4)
    _make_results_tree(tmp_path, {
        "expA": {"loss": rng.normal(size=20)},
        "expB": {"loss": rng.normal(size=20)},
        "expC": {"loss": rng.normal(size=20)},
    })

    rc = _cli(
        str(tmp_path), pattern=_PATTERN, key="loss",
        experiments=["expA", "expC"])

    assert rc == 0
    df = pd.read_csv(StringIO(capsys.readouterr().out), index_col="name")
    assert sorted(df.index) == ["expA", "expC"]
