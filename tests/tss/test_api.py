"""Tests for `nrdk.tss.api`."""

import os

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from nrdk.tss import api
from nrdk.tss.stats import NDStats


def _make_index(tmp_path, data):
    """Write `<exp>/eval/<seq>.npz` fixture files and return an index dict.

    Args:
        tmp_path: root directory (typically pytest's `tmp_path` fixture).
        data: mapping `{experiment: {sequence: {array_name: array, ...}}}`.

    Returns:
        A 2-level index dict `{experiment: {sequence: path}}`, matching the
        shape returned by `api.index()`.
    """
    index = {}
    for exp, sequences in data.items():
        index[exp] = {}
        for seq, arrays in sequences.items():
            path = tmp_path / exp / "eval" / f"{seq}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, **arrays)
            index[exp][seq] = str(path)
    return index


# index()


def test_index_matches_pattern_and_ignores_non_matching_files(tmp_path):
    """A pattern with `experiment`/`trace` groups finds nested `.npz` files.

    Non-matching files/dirs are skipped; sibling dirs are still walked.
    """
    for exp in ["expA", "expB"]:
        for seq in ["seq1", "seq2"]:
            p = tmp_path / exp / "eval" / f"{seq}.npz"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"")
    (tmp_path / "expA" / "eval" / "readme.txt").write_bytes(b"")
    (tmp_path / "notes.md").write_bytes(b"")

    result = api.index(
        str(tmp_path), r"^(?P<experiment>(.*))/eval/(?P<trace>(.*))\.npz$")

    assert result == {
        "expA": {
            "seq1": str(tmp_path / "expA" / "eval" / "seq1.npz"),
            "seq2": str(tmp_path / "expA" / "eval" / "seq2.npz"),
        },
        "expB": {
            "seq1": str(tmp_path / "expB" / "eval" / "seq1.npz"),
            "seq2": str(tmp_path / "expB" / "eval" / "seq2.npz"),
        },
    }


def test_index_pattern_without_trace_group_defaults_trace_to_none(tmp_path):
    """A pattern with only an `experiment` group leaves `trace` as `None`."""
    (tmp_path / "expA.npz").write_bytes(b"")
    (tmp_path / "expB.npz").write_bytes(b"")

    result = api.index(str(tmp_path), r"^(?P<experiment>(.*))\.npz$")

    assert result == {
        "expA": {None: str(tmp_path / "expA.npz")},
        "expB": {None: str(tmp_path / "expB.npz")},
    }


def test_index_symlinks_only_followed_when_requested(tmp_path):
    """`follow_symlinks` controls whether symlinked directories are walked."""
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    (real_dir / "exp1.npz").write_bytes(b"")
    link_dir = tmp_path / "linked"
    os.symlink(real_dir, link_dir)

    pattern = r"^(?P<experiment>(.*))\.npz$"
    no_follow = api.index(str(tmp_path), pattern, follow_symlinks=False)
    follow = api.index(str(tmp_path), pattern, follow_symlinks=True)

    assert "linked/exp1" not in no_follow
    assert "real/exp1" in no_follow
    assert "linked/exp1" in follow
    assert "real/exp1" in follow


def test_experiments_from_index_basic_and_with_timestamps(tmp_path):
    """Load with `key` only: values grouped by experiment, no timestamps.

    Adding `timestamps` loads a parallel per-sequence timestamp array.
    """
    index = _make_index(tmp_path, {
        "expA": {
            "seq1": {"y": np.array([1.0, 2.0]), "t": np.array([0.0, 1.0])},
            "seq2": {"y": np.array([3.0, 4.0]), "t": np.array([2.0, 3.0])},
        },
        "expB": {
            "seq1": {"y": np.array([5.0, 6.0]), "t": np.array([0.0, 1.0])},
            "seq2": {"y": np.array([7.0, 8.0]), "t": np.array([2.0, 3.0])},
        },
    })

    y, t, common = api.experiments_from_index(index, key="y", workers=0)

    assert t is None
    assert sorted(common, key=str) == ["seq1", "seq2"]
    assert sorted(y.keys()) == ["expA", "expB"]
    assert len(y["expA"]) == 2 and len(y["expB"]) == 2
    all_a = np.concatenate(y["expA"])
    all_b = np.concatenate(y["expB"])
    assert sorted(all_a.tolist()) == [1.0, 2.0, 3.0, 4.0]
    assert sorted(all_b.tolist()) == [5.0, 6.0, 7.0, 8.0]

    y2, t2, common2 = api.experiments_from_index(
        index, key="y", timestamps="t", workers=0)
    assert t2 is not None

    assert sorted(common2, key=str) == ["seq1", "seq2"]
    # `common`'s sequence order isn't guaranteed, so compare paired (y, t)
    # values as an order-independent set.
    pairs_a = {
        (tuple(y_arr), tuple(t_arr))
        for y_arr, t_arr in zip(y2["expA"], t2["expA"])}
    assert pairs_a == {((1.0, 2.0), (0.0, 1.0)), ((3.0, 4.0), (2.0, 3.0))}


def test_experiments_from_index_only_common_sequences_loaded(tmp_path):
    """Only sequence names present in *every* requested experiment are kept."""
    index = _make_index(tmp_path, {
        "expA": {
            "seq1": {"y": np.array([1.0])},
            "seq2": {"y": np.array([2.0])},
        },
        "expB": {
            "seq1": {"y": np.array([3.0])},
            # seq2 missing from expB
        },
    })

    y, t, common = api.experiments_from_index(index, key="y", workers=0)

    assert common == ["seq1"]
    assert len(y["expA"]) == 1
    assert len(y["expB"]) == 1


def test_experiments_from_index_cut_splits_on_gap(tmp_path):
    """`cut` splits a single sequence into sub-traces at large gaps."""
    t = np.concatenate(
        [np.arange(5, dtype=float), np.arange(5, dtype=float) + 50])
    y = np.arange(10, dtype=float)
    index = _make_index(tmp_path, {"expA": {"seq1": {"y": y, "t": t}}})

    yy, tt, common = api.experiments_from_index(
        index, key="y", timestamps="t", cut=20.0, workers=0)
    assert tt is not None

    assert len(yy["expA"]) == 2
    assert np.array_equal(yy["expA"][0], np.array([0.0, 1.0, 2.0, 3.0, 4.0]))
    assert np.array_equal(yy["expA"][1], np.array([5.0, 6.0, 7.0, 8.0, 9.0]))
    assert np.array_equal(tt["expA"][0], np.array([0.0, 1.0, 2.0, 3.0, 4.0]))
    assert np.array_equal(
        tt["expA"][1], np.array([50.0, 51.0, 52.0, 53.0, 54.0]))


def test_experiments_from_index_multidim_timestamps_use_last_column(
        tmp_path):
    """Multi-dimensional timestamps are reduced to their last column.

    This holds regardless of whether `cut` is used, so `timestamps` has a
    consistent shape either way.
    """
    t = np.stack([np.zeros(4), np.arange(4, dtype=float)], axis=-1)
    y = np.arange(4, dtype=float)
    index = _make_index(tmp_path, {"expA": {"seq1": {"y": y, "t": t}}})

    yy, tt, common = api.experiments_from_index(
        index, key="y", timestamps="t", cut=None, workers=0)
    assert tt is not None

    seq_t = tt["expA"][0]
    assert isinstance(seq_t, np.ndarray)
    assert seq_t.shape == (4,)
    assert np.array_equal(seq_t, t[:, -1])


def test_experiments_from_index_workers_serial_matches_parallel(tmp_path):
    """`workers=0` and the default `workers=-1` load identical data.

    `workers=0` (serial) and the default (`workers=-1`, parallel) load
    identical data from the same index -- an integration-style
    cross-check between the two code paths.
    """
    rng = np.random.default_rng(0)
    data = {
        exp: {
            seq: {
                "y": rng.normal(size=20),
                "t": np.arange(20, dtype=float),
            }
            for seq in ["seq1", "seq2"]
        }
        for exp in ["expA", "expB", "expC"]
    }
    index = _make_index(tmp_path, data)

    y0, t0, c0 = api.experiments_from_index(
        index, key="y", timestamps="t", workers=0)
    y1, t1, c1 = api.experiments_from_index(
        index, key="y", timestamps="t", workers=-1)

    assert sorted(c0, key=str) == sorted(c1, key=str)
    assert sorted(y0.keys()) == sorted(y1.keys())
    for exp in y0:
        # Thread pool load order may not match; compare as sorted sets of
        # concatenated values to be robust to ordering.
        concat0 = np.sort(np.concatenate(y0[exp]))
        concat1 = np.sort(np.concatenate(y1[exp]))
        assert np.array_equal(concat0, concat1)


@pytest.mark.parametrize("experiments,expected", [
    (r"expA.*", ["expA1", "expA2"]),
    (["expA1", "expB1"], ["expA1", "expB1"]),
])
def test_experiments_from_index_experiment_filter(
        tmp_path, experiments, expected):
    """The `experiments` filter accepts either a regex or an explicit list."""
    index = _make_index(tmp_path, {
        "expA1": {"seq1": {"y": np.array([1.0])}},
        "expA2": {"seq1": {"y": np.array([2.0])}},
        "expB1": {"seq1": {"y": np.array([3.0])}},
    })

    y, t, common = api.experiments_from_index(
        index, key="y", experiments=experiments)

    assert sorted(y.keys()) == sorted(expected)


def test_experiments_from_index_raises_on_empty_or_no_match(tmp_path):
    """An empty index, or an `experiments` filter matching nothing, raises."""
    with pytest.raises(ValueError, match="index is empty"):
        api.experiments_from_index({}, key="y")

    index = _make_index(tmp_path, {"expA": {"seq1": {"y": np.array([1.0])}}})
    with pytest.raises(ValueError, match="No experiments found"):
        api.experiments_from_index(index, key="y", experiments=r"nomatch.*")


def test_experiments_from_index_pattern_without_trace_group(tmp_path):
    """Check a pattern lacking a `trace` group.

    As in the module docstring's flat `<experiment>.npz` example, the
    resulting `common` list is `[None]`.
    """
    np.savez(tmp_path / "expA.npz", y=np.array([1.0, 2.0]))
    index = api.index(str(tmp_path), r"^(?P<experiment>(.*))\.npz$")
    assert index["expA"] == {None: str(tmp_path / "expA.npz")}

    y, t, common = api.experiments_from_index(index, key="y", workers=0)

    assert common == [None]
    assert np.array_equal(y["expA"][0], np.array([1.0, 2.0]))


def test_stats_from_experiments_analytic_mean_std():
    """Mean/std for synthetic per-experiment arrays match direct numpy calc."""
    # NOTE: arrays need to be reasonably large -- `autocorrelation` (used
    # internally to compute ESS) divides by zero for very short sequences.
    a = np.arange(20, dtype=float)
    b = np.arange(30, dtype=float) * 10.0
    y = {"a": [a], "b": [b]}

    names, stats_abs, stats_rel = api.stats_from_experiments(y)

    assert names == ["a", "b"]
    assert stats_rel is None
    assert stats_abs.shape == (2, 1)
    assert stats_abs.mean[:, 0] == pytest.approx(  # type: ignore
        [a.mean(), b.mean()])
    assert stats_abs.std[:, 0] == pytest.approx(  # type: ignore
        [a.std(ddof=1), b.std(ddof=1)])


def test_stats_from_experiments_baseline_relative_without_timestamps():
    """Without timestamps, relative stats are the plain diff vs. baseline."""
    a = np.arange(20, dtype=float)
    b = a + 5.0
    y = {"a": [a], "b": [b]}

    names, stats_abs, stats_rel = api.stats_from_experiments(y, baseline="a")

    assert names == ["a", "b"]
    assert stats_rel is not None
    # Baseline relative to itself is all zeros -> mean 0.
    assert stats_rel.mean[0, 0] == pytest.approx(0.0)  # type: ignore
    # `b` relative to `a` is a constant +5 offset -> mean 5, std 0.
    assert stats_rel.mean[1, 0] == pytest.approx(5.0)  # type: ignore
    assert stats_rel.std[1, 0] == pytest.approx(0.0, abs=1e-8)  # type: ignore


def test_stats_from_experiments_baseline_relative_with_timestamps():
    """With timestamps, relative stats are computed at common timestamps."""
    t_a = np.arange(20, dtype=float)
    y_a = t_a * 10.0
    t_b = np.arange(1, 21, dtype=float)  # offset by 1 -> overlap is t=1..19
    y_b = t_b * 100.0
    y = {"a": [y_a], "b": [y_b]}
    t = {"a": [t_a], "b": [t_b]}

    names, stats_abs, stats_rel = api.stats_from_experiments(y, t, baseline="a")
    assert stats_rel is not None

    # Common timestamps between a and b: {1, ..., 19}.
    common_t = np.arange(1, 20, dtype=float)
    expected_diff_b = common_t * 100.0 - common_t * 10.0
    assert stats_rel.mean[1, 0] == pytest.approx(  # type: ignore
        expected_diff_b.mean())
    assert stats_rel.n[1, 0] == expected_diff_b.shape[0]  # type: ignore


def test_stats_from_experiments_mismatched_keys_raises_value_error():
    """Different key sets in `y` and `t` raise `ValueError`."""
    y = {"a": [np.array([1.0, 2.0])], "b": [np.array([3.0, 4.0])]}
    t = {"a": [np.array([0.0, 1.0])]}

    with pytest.raises(ValueError, match="Keys of `y` and `t` must match"):
        api.stats_from_experiments(y, t)


def test_dataframe_from_stats_abs_only_columns():
    """Without `rel`, only `abs/*` columns are produced."""
    rng = np.random.default_rng(21)
    names = ["a", "b"]
    arrays = [rng.normal(size=30) + i for i in range(2)]
    stats_abs = NDStats.from_values(arrays)

    df = api.dataframe_from_stats(names, stats_abs)

    assert list(df.columns) == [
        "abs/mean", "abs/std", "abs/stderr", "abs/n", "abs/ess"]
    assert list(df.index) == names


def test_dataframe_from_stats_pct_and_p_value_formulas():
    """`pct/mean`, `pct/stderr`, and `p0.05` match their documented formulas."""
    rng = np.random.default_rng(22)
    names = ["base", "x", "y"]
    arrays = [rng.normal(size=200) + i * 2 for i in range(3)]
    stats_abs = NDStats.from_values(arrays)
    # Build "relative" stats independently (difference vs. the first array).
    rel_arrays = [arr - arrays[0] for arr in arrays]
    stats_rel = NDStats.from_values(rel_arrays)

    df = api.dataframe_from_stats(names, stats_abs, stats_rel, baseline="base")

    base_abs_mean = df.loc["base", "abs/mean"]  # type: ignore
    expected_pct_mean = df["rel/mean"] / base_abs_mean * 100  # type: ignore
    expected_pct_stderr = df["rel/stderr"] / base_abs_mean * 100  # type: ignore
    z = norm.ppf(1 - 0.05 / 2 / (len(names) - 1))
    expected_p = (df["rel/mean"] / df["rel/stderr"]) > z

    # NOTE: the baseline row's `rel/stderr` is `0/sqrt(0) = nan` (its own
    # diff-from-itself is a constant-zero array, whose ESS is reported as
    # 0.0), so `pct/stderr` and the `p0.05` comparison are `nan` there too;
    # `equal_nan=True` treats matching nans as equal on both sides.
    np.testing.assert_allclose(
        df["pct/mean"].to_numpy(), expected_pct_mean.to_numpy(), equal_nan=True)
    np.testing.assert_allclose(
        df["pct/stderr"].to_numpy(), expected_pct_stderr.to_numpy(),
        equal_nan=True)
    assert df["p0.05"].tolist() == expected_p.tolist()


def test_dataframe_from_stats_rel_without_baseline_raises_value_error():
    """Providing `rel` without a `baseline` name raises `ValueError`."""
    rng = np.random.default_rng(23)
    names = ["a", "b"]
    arrays = [rng.normal(size=30) + i for i in range(2)]
    stats_abs = NDStats.from_values(arrays)
    stats_rel = NDStats.from_values(arrays)

    with pytest.raises(ValueError, match="baseline"):
        api.dataframe_from_stats(names, stats_abs, stats_rel, baseline=None)


def test_dataframe_from_index_end_to_end(tmp_path):
    """Chains index -> experiments -> stats -> dataframe helpers.

    Chains `experiments_from_index` -> `stats_from_experiments` ->
    `dataframe_from_stats` and checks the result against a manual
    computation of the same pipeline.
    """
    rng = np.random.default_rng(24)
    data = {
        "base": {
            "seq1": {
                "y": rng.normal(size=50) + 1.0,
                "t": np.arange(50, dtype=float),
            },
        },
        "treatment": {
            "seq1": {
                "y": rng.normal(size=50) + 3.0,
                "t": np.arange(50, dtype=float),
            },
        },
    }
    index = _make_index(tmp_path, data)

    df = api.dataframe_from_index(
        index, key="y", timestamps="t", baseline="base", workers=0)

    y, t, _ = api.experiments_from_index(
        index, key="y", timestamps="t", workers=0)
    names, stats_abs, stats_rel = api.stats_from_experiments(
        y, t, baseline="base")
    expected_df = api.dataframe_from_stats(
        names, stats_abs, stats_rel, baseline="base")

    assert list(df.columns) == list(expected_df.columns)
    pd.testing.assert_frame_equal(df, expected_df)
    assert set(df.index) == {"base", "treatment"}
    # Sanity: treatment's absolute mean should be notably higher than base's.
    assert (
        df.loc["treatment", "abs/mean"]  # type: ignore
        > df.loc["base", "abs/mean"])
