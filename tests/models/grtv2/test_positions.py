"""Tests for GRTv2 positional encoders."""

import pytest
import torch

from nrdk.models.grtv2 import (
    InputPositionalEncoder,
    OutputPositionalEncoder,
    Query,
)

D_MODEL = 16
N, T = 2, 3
D_TOK, R_TOK = 4, 8


def _timestamps():
    return torch.arange(T, dtype=torch.float32)[None].repeat(N, 1) * 0.1


def _query(n=N, q=6, **overrides):
    torch.manual_seed(1)
    fields = {k: torch.randn(n, q) for k in ("t", "rng", "az", "el")}
    fields.update(overrides)
    return Query(**fields)


def test_input_positional_encoder_shapes():
    """Spatial tokens and per-frame tokens both preserve their shape."""
    pe = InputPositionalEncoder(d_model=D_MODEL, d_features=8)
    t = _timestamps()
    x5 = torch.randn(N, T, D_TOK, R_TOK, D_MODEL)
    x3 = torch.randn(N, T, D_MODEL)
    assert pe(x5, t).shape == x5.shape
    assert pe(x3, t).shape == x3.shape


def test_output_positional_encoder_shape():
    """Each query yields one `d_model` position token."""
    pe = OutputPositionalEncoder(d_model=D_MODEL, d_features=8)
    assert pe(_query()).shape == (N, 6, D_MODEL)


def test_output_positional_encoder_omits_unset_axes():
    """Unset axes without wildcards contribute nothing."""
    pe = OutputPositionalEncoder(
        d_model=D_MODEL, d_features=8, axes=("t", "rng"))
    full = _query(az=None, el=None)
    expected = (
        pe.project["t"](pe.pos(full.t * pe.scale_t))  # type: ignore
        + pe.project["rng"](pe.pos(full.rng)))  # type: ignore
    assert torch.allclose(pe(full), expected)


def test_output_positional_encoder_wildcard():
    """Unset axes with a wildcard use the learned wildcard token."""
    pe = OutputPositionalEncoder(
        d_model=D_MODEL, d_features=8, axes=("t",), wildcards=("rng",))
    q = _query(rng=None, az=None, el=None)
    without = OutputPositionalEncoder(
        d_model=D_MODEL, d_features=8, axes=("t",))
    without.load_state_dict(pe.state_dict(), strict=False)
    assert torch.allclose(
        pe(q), without(q) + pe.wildcards["rng"][None, None, :])


def test_output_positional_encoder_empty():
    """With no contributing axes, the output is zero."""
    pe = OutputPositionalEncoder(d_model=D_MODEL, d_features=8, axes=())
    out = pe(Query(t=None, rng=None, az=None, el=None))
    assert torch.all(out == 0)


def test_output_positional_encoder_unconfigured_axis_raises():
    """Providing an axis which is not in `axes` is an error."""
    pe = OutputPositionalEncoder(d_model=D_MODEL, d_features=8, axes=("t",))
    with pytest.raises(ValueError):
        pe(_query())


def test_output_positional_encoder_invalid_axis_key_raises():
    """Unknown axis names are rejected at construction."""
    with pytest.raises(Exception):
        OutputPositionalEncoder(axes=("t", "bogus"))  # type: ignore
