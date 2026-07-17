"""Tests for `nrdk.modules.position`."""

import math

import pytest
import torch

from nrdk.modules.position import (
    BasisChange,
    FourierFeatures,
    LearnableND,
    Readout,
    Sinusoid,
)


def test_sinusoid_preserves_shape():
    """Sinusoid embeddings are added elementwise; shape is unchanged."""
    x = torch.randn(2, 5, 8)
    out = Sinusoid(scale=1.0, w_min=1.0)(x)
    assert out.shape == x.shape


def test_sinusoid_identity_at_center_position():
    """At the centered position (`t=0`), `sin(0)=0` and `cos(0)=1`.

    With an odd-length spatial axis, the default linspace(-1, 1, n) passes
    exactly through 0 at the center index, regardless of `w`.
    """
    x = torch.zeros(1, 5, 8)
    out = Sinusoid(scale=1.0, w_min=1.0)(x)

    center = out[0, 2]  # linspace(-1, 1, 5)[2] == 0.0
    expected = torch.tensor([0.0, 1.0] * 4)
    assert torch.allclose(center, expected, atol=1e-6)


def test_sinusoid_explicit_positions_override_default():
    """Explicit `positions` are used (and scaled) instead of the default."""
    x = torch.zeros(1, 3, 4)
    positions = [torch.tensor([[0.0, 0.0, 0.0]])]

    out = Sinusoid(scale=1.0, w_min=1.0)(x, positions=positions)
    expected = torch.tensor([0.0, 1.0, 0.0, 1.0]).expand(1, 3, 4)
    assert torch.allclose(out, expected, atol=1e-6)


def test_sinusoid_positions_length_mismatch_raises():
    """Wrong number of position sequences raises `ValueError`."""
    x = torch.zeros(2, 5, 6, 8)  # 2 spatial axes
    with pytest.raises(ValueError):
        Sinusoid()(x, positions=[torch.zeros(1, 5)])


def test_sinusoid_channels_length_mismatch_raises():
    """`channels` must specify one entry per spatial axis."""
    x = torch.zeros(2, 5, 6, 8)  # 2 spatial axes
    with pytest.raises(ValueError):
        Sinusoid(channels=[2])(x)


def test_sinusoid_channels_too_large_raises():
    """`channels` requesting more than the available channels raises."""
    x = torch.zeros(2, 5, 6, 8)
    with pytest.raises(ValueError):
        Sinusoid(channels=[10, 10])(x)


def test_sinusoid_scale_length_mismatch_raises():
    """`scale` must specify one entry per spatial axis."""
    x = torch.zeros(2, 5, 6, 8)
    with pytest.raises(ValueError):
        Sinusoid(scale=[1.0])(x)


def test_sinusoid_w_min_length_mismatch_raises():
    """`w_min` must specify one entry per spatial axis."""
    x = torch.zeros(2, 5, 6, 8)
    with pytest.raises(ValueError):
        Sinusoid(w_min=[1.0])(x)


# LearnableND


def test_learnable_nd_matches_broadcast_sum_of_embeddings():
    """The output is the elementwise sum of each axis's learned embedding."""
    torch.manual_seed(0)
    ln = LearnableND(d_model=4, shape=(2, 3))
    x = torch.zeros(1, 2, 3, 4)

    out = ln(x)
    e0, e1 = ln.embeddings[0], ln.embeddings[1]  # (2, 4), (3, 4)
    expected = e0[:, None, :] + e1[None, :, :]
    assert torch.allclose(out[0], expected)


def test_learnable_nd_adds_to_existing_input():
    """Nonzero input is preserved under the additive embedding."""
    torch.manual_seed(0)
    ln = LearnableND(d_model=4, shape=(2, 3))
    x = torch.randn(2, 2, 3, 4)

    out = ln(x)
    e0, e1 = ln.embeddings[0], ln.embeddings[1]
    expected = x + e0[:, None, :] + e1[None, :, :]
    assert torch.allclose(out, expected)


# Readout


def test_readout_appends_token_and_preserves_prefix():
    """The readout token is tiled and appended as the last sequence entry."""
    torch.manual_seed(0)
    readout = Readout(d_model=4)
    x = torch.randn(2, 5, 4)

    out = readout(x)
    assert out.shape == (2, 6, 4)
    assert torch.allclose(out[:, :-1], x)
    assert torch.allclose(out[0, -1], readout.readout)
    assert torch.allclose(out[1, -1], readout.readout)


# BasisChange


def test_basis_change_flatten_shape():
    """`flatten=True` collapses the query shape into a single axis."""
    torch.manual_seed(0)
    bc = BasisChange(shape=[2, 3], flatten=True, scale=1.0, w_min=1.0)
    x = torch.randn(2, 4)

    out = bc(x)
    assert out.shape == (2, 6, 4)


def test_basis_change_matches_manual_tile_and_sinusoid():
    """Output equals the reference vector tiled, plus a sinusoid embedding.

    With `scale=1.0`, `w_min=1.0`, and 1 channel pair per axis (since
    `d_model=4` splits evenly 2-and-2 across 2 axes), each output channel is
    hand-computable: channels 0/1 carry axis-0's sin/cos, channels 2/3 carry
    axis-1's sin/cos, each added to the tiled reference vector.
    """
    torch.manual_seed(0)
    bc = BasisChange(shape=[2, 3], flatten=False, scale=1.0, w_min=1.0)
    x = torch.randn(2, 4)

    out = bc(x)
    assert out.shape == (2, 2, 3, 4)

    ti = torch.linspace(-1.0, 1.0, 2)
    tj = torch.linspace(-1.0, 1.0, 3)
    expected = torch.zeros(2, 2, 3, 4)
    for b in range(2):
        for i in range(2):
            for j in range(3):
                expected[b, i, j, 0] = x[b, 0] + math.sin(math.pi * ti[i])
                expected[b, i, j, 1] = x[b, 1] + math.cos(math.pi * ti[i])
                expected[b, i, j, 2] = x[b, 2] + math.sin(math.pi * tj[j])
                expected[b, i, j, 3] = x[b, 3] + math.cos(math.pi * tj[j])

    assert torch.allclose(out, expected, atol=1e-5)


# FourierFeatures


def test_fourier_features_odd_count_raises():
    """An odd `features` count raises `ValueError`."""
    with pytest.raises(ValueError):
        FourierFeatures(features=5)


def test_fourier_features_shape():
    """Fourier features append a trailing features axis."""
    ff = FourierFeatures(features=4, coef=100.0)
    out = ff(torch.randn(3, 5))
    assert out.shape == (3, 5, 4)


def test_fourier_features_identity_at_zero():
    """At `x=0`, `cos(2*pi*w*0)=1` and `sin(2*pi*w*0)=0` for every frequency.

    The first half of the features axis is `cos` and the second half is
    `sin` (see the concatenation order in the source).
    """
    ff = FourierFeatures(features=6, coef=100.0)
    out = ff(torch.zeros(4))

    nc = 3
    assert torch.allclose(out[:, :nc], torch.ones(4, nc), atol=1e-6)
    assert torch.allclose(out[:, nc:], torch.zeros(4, nc), atol=1e-6)
