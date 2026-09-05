"""Tests for `nrdk.modules.scale`."""

import torch

from nrdk.modules.scale import ASinh, Mixed, Power, Rational


def _small_spectrum(b=2, t=1, d=4, tx=2, rx=2, rng=4, ch=2, seed=0):
    """Generate a small random spectrum `(b, t, d, tx, rx, rng, ch)`."""
    torch.manual_seed(seed)
    return torch.randn(b, t, d, tx, rx, rng, ch)


# Power


def test_power_forward_changes_magnitude():
    """Forward scaling with `power != 1` changes the output."""
    x = _small_spectrum()
    out = Power(power=0.5)(x)

    assert out.shape == x.shape
    assert torch.all(torch.isfinite(out))
    assert not torch.allclose(out, x)


def test_power_identity_at_one():
    """`power=1` leaves the magnitude unchanged."""
    x = _small_spectrum()
    out = Power(power=1.0)(x)

    assert torch.allclose(out, x, atol=1e-5)


def test_power_roundtrip():
    """Forward then reverse with the same power recovers the input."""
    x = _small_spectrum()
    scale = Power(power=0.5)

    recovered = scale(scale(x), reverse=True)
    assert torch.allclose(recovered, x, atol=1e-4)


def test_power_zero_input_is_finite():
    """All-zero input does not produce NaN/Inf after `nan_to_num`."""
    x = torch.zeros(2, 1, 4, 2, 2, 4, 2)
    out = Power(power=0.5)(x)

    assert torch.all(torch.isfinite(out))


# Rational


def test_rational_roundtrip():
    """Forward then reverse with the same coefficient recovers the input."""
    x = _small_spectrum()
    scale = Rational(coef=0.05)

    recovered = scale(scale(x), reverse=True)
    assert torch.allclose(recovered, x, atol=1e-4)


def test_rational_zero_input_is_finite():
    """All-zero input does not produce NaN/Inf after `nan_to_num`."""
    x = torch.zeros(2, 1, 4, 2, 2, 4, 2)
    out = Rational(coef=0.05)(x)

    assert torch.all(torch.isfinite(out))


# ASinh


def test_asinh_roundtrip():
    """Forward then reverse with the same scale recovers the input."""
    x = _small_spectrum()
    scale = ASinh(scale=1.0)

    recovered = scale(scale(x), reverse=True)
    assert torch.allclose(recovered, x, atol=1e-3)


def test_asinh_compresses_large_magnitudes():
    """Large magnitudes are compressed relative to the input."""
    x = _small_spectrum() * 100
    out = ASinh(scale=1.0)(x)

    rd_in = torch.linalg.norm(
        x.reshape(*x.shape[:3], x.shape[5], -1), dim=-1)
    rd_out = torch.linalg.norm(
        out.reshape(*out.shape[:3], out.shape[5], -1), dim=-1)
    assert torch.all(rd_out < rd_in)


def test_asinh_zero_input_is_finite():
    """All-zero input does not produce NaN/Inf after `nan_to_num`."""
    x = torch.zeros(2, 1, 4, 2, 2, 4, 2)
    out = ASinh(scale=1.0)(x)

    assert torch.all(torch.isfinite(out))


# Mixed


def test_mixed_uses_forward_scale_in_forward_direction():
    """Forward calls delegate to `forward_scale`."""
    x = _small_spectrum()
    mixed = Mixed(forward=Power(power=0.5), reverse=Rational(coef=0.05))

    assert torch.allclose(mixed(x), Power(power=0.5)(x))


def test_mixed_uses_reverse_scale_in_reverse_direction():
    """`reverse=True` calls `reverse_scale`, itself in its forward direction."""
    x = _small_spectrum()
    mixed = Mixed(forward=Power(power=0.5), reverse=Rational(coef=0.05))

    expected = Rational(coef=0.05)(x)
    assert torch.allclose(mixed(x, reverse=True), expected)


def test_mixed_none_forward_is_identity():
    """`forward=None` passes the input through unchanged."""
    x = _small_spectrum()
    mixed = Mixed(forward=None, reverse=Power(power=0.5))

    assert torch.equal(mixed(x), x)


def test_mixed_none_reverse_is_identity():
    """`reverse=None` passes the input through unchanged."""
    x = _small_spectrum()
    mixed = Mixed(forward=Power(power=0.5), reverse=None)

    assert torch.equal(mixed(x, reverse=True), x)
