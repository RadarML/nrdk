"""Tests for GRT tokenizers."""

import pytest
import torch
from abstract_dataloader.ext import types
from jaxtyping import Float
from torch import Tensor

from nrdk import models
from nrdk.models.grt import SpectrumAngularMLPTokenizer, SpectrumTokenizer


@types.dataclass
class _Spectrum:
    """Minimal container satisfying the `nrdk.types.SpectrumData` protocol."""

    spectrum: Float[Tensor, "batch t doppler x1 x2 rng ch"]
    timestamps: Float[Tensor, "batch t"]


def _spectrum(n=2, t=1, doppler=4, el=2, az=4, rng=8, ch=2, seed=0):
    """Build a small `(batch t doppler el az rng ch)` spectrum tensor."""
    torch.manual_seed(seed)
    return torch.randn(n, t, doppler, el, az, rng, ch)


def _wrap(x):
    """Wrap a raw spectrum tensor in a `SpectrumData`-like container."""
    return _Spectrum(
        spectrum=x, timestamps=torch.zeros(x.shape[0], x.shape[1]))


# SpectrumTokenizer


@pytest.mark.parametrize("positions", ["nd", "flat"])
def test_spectrum_tokenizer_output_shape(positions):
    """Output has one token per patch, plus a trailing readout token."""
    tokenizer = SpectrumTokenizer(
        d_model=20, patch=(1, 2, 2, 2, 4), n_channels=2, positions=positions)
    out = tokenizer(_spectrum())
    # (1 * 2 * 1 * 2 * 2) patches + 1 readout
    assert out.shape == (2, 9, 20)
    assert torch.all(torch.isfinite(out))


def test_spectrum_tokenizer_squeeze_output_shape():
    """Squeezed axes are moved to channels, removing them from the tokens."""
    tokenizer = SpectrumTokenizer(
        d_model=16, patch=(1, 2, 2, 2, 4), squeeze=[2], n_channels=2)
    out = tokenizer(_spectrum())
    # (1 * 2 * 2 * 2) patches + 1 readout
    assert out.shape == (2, 9, 16)


def test_spectrum_tokenizer_accepts_spectrum_data_or_tensor():
    """`SpectrumData` and raw tensor inputs produce identical tokens."""
    tokenizer = SpectrumTokenizer(
        d_model=20, patch=(1, 2, 2, 2, 4), n_channels=2)
    x = _spectrum()
    assert torch.allclose(tokenizer(x), tokenizer(_wrap(x)))


def test_spectrum_tokenizer_invalid_patch_raises():
    """A patch size without exactly 5 axes is rejected."""
    with pytest.raises(ValueError):
        SpectrumTokenizer(patch=(2, 2, 2, 2))


# SpectrumAngularMLPTokenizer


def _angular(**kwargs):
    return SpectrumAngularMLPTokenizer(
        d_model=12, mlp=[8, 8], patch=(1, 2, 4), angle_bins=(2, 4),
        n_channels=2, **kwargs)


@pytest.mark.parametrize("positions", ["nd", "flat"])
def test_spectrum_angular_mlp_tokenizer_output_shape(positions):
    """Angular axes are squeezed; tokens are cut over (time, doppler, range)."""
    out = _angular(positions=positions)(_spectrum())
    # (1 * 2 * 2) patches + 1 readout
    assert out.shape == (2, 5, 12)
    assert torch.all(torch.isfinite(out))


def test_spectrum_angular_mlp_tokenizer_accepts_spectrum_data_or_tensor():
    """`SpectrumData` and raw tensor inputs produce identical tokens."""
    tokenizer = _angular()
    x = _spectrum()
    assert torch.allclose(tokenizer(x), tokenizer(_wrap(x)))


def test_spectrum_angular_mlp_tokenizer_invalid_patch_raises():
    """A patch size without exactly 3 axes is rejected."""
    with pytest.raises(ValueError):
        SpectrumAngularMLPTokenizer(patch=(1, 2, 2, 4))


# Deprecated aliases


def test_top_level_aliases():
    """Old `nrdk.models.*` targets resolve to the `nrdk.models.grt` classes."""
    assert models.TransformerTensorDecoder is \
        models.grt.TransformerTensorDecoder
    assert models.MLPVectorDecoder is models.grt.MLPVectorDecoder
