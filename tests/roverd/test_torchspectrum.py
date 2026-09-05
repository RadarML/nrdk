"""Tests for `nrdk.roverd.torchspectrum`."""

import pytest
import torch
from roverd import types
from xwr.nn import Magnitude
from xwr.rsp.torch import AWR1843Boost

from nrdk.roverd import TorchSpectrum


def _small_iq(b=2, t=1, slow=4, tx=3, rx=4, fast=16, seed=0):
    """Generate small int16 IIQQ-interleaved I/Q data.

    AWR1843Boost's virtual array requires exactly `tx=3, rx=4`.
    """
    torch.manual_seed(seed)
    iq = torch.randint(
        -100, 100, (b, t, slow, tx, rx, fast), dtype=torch.int16)
    return types.XWRRadarIQ(
        iq=iq,
        timestamps=torch.arange(b * t, dtype=torch.float64).reshape(b, t),
        range_resolution=torch.full((b,), 0.1),
        doppler_resolution=torch.full((b,), 0.05),
        valid=torch.ones((b, t), dtype=torch.uint8))


def test_spectrum_full_mode_shape():
    """Full mode produces a spectrum with metadata carried through."""
    iq = _small_iq()
    transform = TorchSpectrum(
        rsp=AWR1843Boost(), rep=Magnitude(), mode="full")

    out = transform(iq)

    assert out.spectrum.shape[0] == 2  # batch
    assert out.spectrum.shape[1] == 1  # t
    assert out.spectrum.shape[-1] == 1  # Magnitude -> single channel
    assert torch.all(torch.isfinite(out.spectrum))
    assert torch.equal(out.timestamps, iq.timestamps)
    assert torch.equal(out.range_resolution, iq.range_resolution)


def test_spectrum_rd_mode_shape():
    """RD mode produces a range-Doppler-only spectrum (no angular axes)."""
    iq = _small_iq()
    transform = TorchSpectrum(rsp=AWR1843Boost(), rep=Magnitude(), mode="rd")

    out = transform(iq)

    assert out.spectrum.shape[0] == 2
    assert torch.all(torch.isfinite(out.spectrum))


def test_spectrum_rd_mode_rejects_angular_augmentations():
    """RD mode raises if an angular-only augmentation is requested."""
    iq = _small_iq()
    transform = TorchSpectrum(rsp=AWR1843Boost(), rep=Magnitude(), mode="rd")

    aug = {"azimuth_flip": torch.zeros(2, dtype=torch.bool)}
    with pytest.raises(ValueError):
        transform(iq, aug=aug)
