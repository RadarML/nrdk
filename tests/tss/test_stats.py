"""Tests for `nrdk.tss.stats`."""

import numpy as np
import pandas as pd
import pytest

from nrdk.tss.stats import NDStats


def test_ndstats_from_values_stacks_and_recurses():
    """A list of arrays stacks per-array; nested lists recurse to a 2D stack."""
    rng = np.random.default_rng(11)
    arrays = [rng.normal(size=30) + i for i in range(4)]

    stats = NDStats.from_values(arrays)

    assert stats.shape == (4,)
    for i, arr in enumerate(arrays):
        assert stats.mean[i] == pytest.approx(arr.mean())  # type: ignore
        assert stats.std[i] == pytest.approx(arr.std(ddof=1))  # type: ignore
        assert stats.n[i] == arr.shape[0]  # type: ignore

    rng2 = np.random.default_rng(12)
    nested = [
        [rng2.normal(size=25) + i + j for j in range(3)] for i in range(2)
    ]

    nested_stats = NDStats.from_values(nested, workers=0)

    assert nested_stats.shape == (2, 3)
    expected_mean = np.array(
        [[v.mean() for v in row] for row in nested])
    assert np.allclose(nested_stats.mean, expected_mean)


def test_ndstats_from_values_serial_matches_parallel():
    """`workers=0` (serial) and `workers=-1` (parallel) give identical stats.

    Integration-style cross-check: the two code paths must agree exactly on
    the same input, matching the "compressed vs. direct" pattern used
    elsewhere for cross-validating a fast/parallel path against a serial one.
    """
    rng = np.random.default_rng(13)
    nested = [
        [rng.normal(size=25) + i + j for j in range(2)] for i in range(3)
    ]

    stats_serial = NDStats.from_values(nested, workers=0)
    stats_parallel = NDStats.from_values(nested, workers=-1)

    assert np.array_equal(stats_serial.n, stats_parallel.n)
    assert np.allclose(stats_serial.m1, stats_parallel.m1)
    assert np.allclose(stats_serial.m2, stats_parallel.m2)
    assert np.allclose(stats_serial.ess, stats_parallel.ess)


def test_ndstats_stack_and_reshape():
    """`stack` concatenates scalar stats; `reshape` reshapes all fields."""
    rng = np.random.default_rng(14)
    a = rng.normal(size=30) + 1.0
    b = rng.normal(size=25) + 2.0

    stacked = NDStats.stack(NDStats.from_values(a), NDStats.from_values(b))

    assert stacked.shape == (2,)
    assert stacked.n.tolist() == [a.shape[0], b.shape[0]]
    assert stacked.mean == pytest.approx([a.mean(), b.mean()])

    rng2 = np.random.default_rng(17)
    stacked6 = NDStats.stack(
        *[NDStats.from_values(rng2.normal(size=15) + i) for i in range(6)])

    reshaped = stacked6.reshape(2, 3)

    assert reshaped.shape == (2, 3)
    assert np.array_equal(reshaped.n, stacked6.n.reshape(2, 3))
    assert np.allclose(reshaped.m1, stacked6.m1.reshape(2, 3))
    assert np.allclose(reshaped.m2, stacked6.m2.reshape(2, 3))
    assert np.allclose(reshaped.ess, stacked6.ess.reshape(2, 3))


def test_ndstats_sum():
    """`sum` aggregates n/m1/m2/ess across an axis; on a scalar it's a no-op."""
    rng = np.random.default_rng(15)
    stacked = NDStats.stack(
        *[NDStats.from_values(rng.normal(size=20) + i) for i in range(6)])
    reshaped = stacked.reshape(2, 3)

    summed = reshaped.sum(axis=-1)

    assert summed.shape == (2,)
    assert np.array_equal(summed.n, np.sum(reshaped.n, axis=-1))
    assert np.allclose(summed.m1, np.sum(reshaped.m1, axis=-1))
    assert np.allclose(summed.m2, np.sum(reshaped.m2, axis=-1))
    assert np.allclose(summed.ess, np.sum(reshaped.ess, axis=-1))

    x = np.random.default_rng(16).normal(size=20)
    scalar_stats = NDStats.from_values(x)

    scalar_summed = scalar_stats.sum()

    assert scalar_summed.n == scalar_stats.n
    assert scalar_summed.m1 == scalar_stats.m1


def test_ndstats_as_df_columns_match_properties():
    """`as_df` columns equal the corresponding scalar properties per row."""
    rng = np.random.default_rng(18)
    names = ["a", "b", "c"]
    arrays = [rng.normal(size=30) + i for i in range(3)]
    stats = NDStats.from_values(arrays)

    df = stats.as_df(names, prefix="abs/")

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == [
        "abs/mean", "abs/std", "abs/stderr", "abs/n", "abs/ess"]
    assert list(df.index) == names
    assert np.allclose(df["abs/mean"].to_numpy(), stats.mean)
    assert np.allclose(df["abs/std"].to_numpy(), stats.std)
    assert np.allclose(df["abs/stderr"].to_numpy(), stats.stderr)
    assert np.array_equal(df["abs/n"].to_numpy(), stats.n)
    assert np.allclose(df["abs/ess"].to_numpy(), stats.ess)


def test_ndstats_as_df_errors():
    """`as_df` raises for a non-1D `NDStats`, or a `names` length mismatch."""
    x = np.random.default_rng(19).normal(size=20)
    scalar_stats = NDStats.from_values(x)
    with pytest.raises(ValueError):
        scalar_stats.as_df(["a"])

    rng = np.random.default_rng(20)
    stats = NDStats.from_values([rng.normal(size=20) + i for i in range(2)])
    with pytest.raises(ValueError):
        stats.as_df(["only_one_name"])
