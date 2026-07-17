"""Tests for `nrdk.tss.utils`."""

import numpy as np
import pytest

from nrdk.tss.utils import (
    cut_trace,
    intersect_difference,
    tree_flatten,
    tree_unflatten,
)


def test_tree_flatten_unflatten_roundtrip():
    """Flattening then unflattening recovers the original nested structure."""
    tree = {
        "a": np.array([1, 2, 3]),
        "b": {
            "c": np.array([4, 5]),
            "d": {"e": np.array([6, 7])},
        },
    }

    flat = tree_flatten(tree)
    back = tree_unflatten(flat)

    assert set(back.keys()) == set(tree.keys())
    assert np.array_equal(back["a"], tree["a"])
    assert np.array_equal(back["b"]["c"], tree["b"]["c"])
    assert np.array_equal(back["b"]["d"]["e"], tree["b"]["d"]["e"])


def test_tree_flatten_uses_slash_joined_paths():
    """Flattened keys are `/`-joined paths; a bare leaf flattens to `""`."""
    tree = {"a": np.array([1]), "b": {"c": np.array([2]), "d": np.array([3])}}

    flat = tree_flatten(tree)

    assert set(flat.keys()) == {"a", "b/c", "b/d"}
    assert flat["b/c"].item() == 2  # type: ignore
    assert flat["b/d"].item() == 3  # type: ignore

    leaf = np.array([1, 2, 3])
    flat_leaf = tree_flatten(leaf)

    assert list(flat_leaf.keys()) == [""]
    assert np.array_equal(flat_leaf[""], leaf)


def test_tree_unflatten_hand_built_key_paths():
    """A hand-built dict of `/`-joined keys unflattens to nested dicts."""
    flat = {"x/y/z": np.array([1]), "x/w": np.array([2]), "top": np.array([3])}

    tree = tree_unflatten(flat)

    assert np.array_equal(tree["x"]["y"]["z"], np.array([1]))
    assert np.array_equal(tree["x"]["w"], np.array([2]))
    assert np.array_equal(tree["top"], np.array([3]))


@pytest.mark.parametrize("n_gaps,expected", [
    (0, [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]),
    (1, [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]),
    (2, [[0, 1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11]]),
])
def test_cut_trace_splits_at_gaps(n_gaps, expected):
    """A trace is split into one piece per gap exceeding the threshold."""
    segment_lengths = {0: [10], 1: [5, 5], 2: [5, 4, 3]}[n_gaps]
    t = np.concatenate([
        np.arange(length, dtype=float) + 50 * i
        for i, length in enumerate(segment_lengths)])
    y = np.arange(t.shape[0])

    pieces = cut_trace(t, y, gap=20.0)

    assert len(pieces) == len(expected)
    for piece, exp in zip(pieces, expected):
        assert np.array_equal(piece, exp)
    # Pieces cover every input index exactly once, in order.
    assert np.array_equal(np.concatenate(pieces), y)


def test_cut_trace_preserves_nested_pytree_structure_per_cut():
    """Each returned sub-trace preserves the nested structure of `values`."""
    t = np.concatenate(
        [np.arange(5, dtype=float), np.arange(4, dtype=float) + 50])
    values = {
        "y": np.arange(t.shape[0]),
        "meta": {"id": np.arange(t.shape[0]) * 10},
    }

    pieces = cut_trace(t, values, gap=20.0)

    assert len(pieces) == 2
    for piece in pieces:
        assert set(piece.keys()) == {"y", "meta"}
        assert set(piece["meta"].keys()) == {"id"}
        # `id` sub-array stays aligned (10x) with the `y` sub-array per cut.
        assert np.array_equal(piece["meta"]["id"], piece["y"] * 10)

    assert np.array_equal(pieces[0]["y"], np.array([0, 1, 2, 3, 4]))
    assert np.array_equal(pieces[1]["y"], np.array([5, 6, 7, 8]))


def test_intersect_difference_without_timestamps():
    """Check `intersect_difference` without timestamps.

    Matched shapes give a plain diff; mismatched shapes raise `ValueError`.
    """
    y1 = np.array([1.0, 2.0, 3.0])
    y2 = np.array([1.0, 1.0, 1.0])

    diff = intersect_difference(y1, y2)

    assert np.array_equal(diff, np.array([0.0, 1.0, 2.0]))

    with pytest.raises(ValueError):
        intersect_difference(y1, np.array([1.0, 2.0]))


def test_intersect_difference_partial_timestamp_overlap():
    """With timestamps, only the common timestamps are diffed, in order."""
    y1 = np.array([10.0, 20.0, 30.0, 40.0])
    t1 = np.array([0.0, 1.0, 2.0, 3.0])
    y2 = np.array([100.0, 200.0, 400.0])
    t2 = np.array([1.0, 2.0, 4.0])

    diff = intersect_difference(y1, y2, t1, t2)

    # Common timestamps are {1, 2}: y1 there is [20, 30], y2 there is
    # [100, 200].
    assert np.array_equal(diff, np.array([20.0 - 100.0, 30.0 - 200.0]))
