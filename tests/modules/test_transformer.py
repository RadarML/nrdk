"""Tests for `nrdk.modules.transformer`."""

import pytest
import torch
from jaxtyping import TypeCheckError

from nrdk.modules.transformer import (
    AATransformerEncoder,
    RMSNorm,
    VisionTransformerEncoderLayer,
)

# RMSNorm


def test_rms_norm_preserves_shape():
    """RMSNorm preserves input shape."""
    torch.manual_seed(0)
    x = torch.randn(2, 5, 16)
    out = RMSNorm(d_model=16)(x)

    assert out.shape == x.shape


def test_rms_norm_normalizes_scale():
    """With unit weight, the RMS of the output is approximately 1."""
    torch.manual_seed(0)
    x = torch.randn(2, 5, 16) * 10
    out = RMSNorm(d_model=16, eps=1e-8)(x)

    rms = torch.sqrt(torch.mean(out ** 2, dim=-1))
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-2)


# VisionTransformerEncoderLayer


def test_vit_encoder_layer_preserves_shape():
    """A single layer preserves the input shape."""
    torch.manual_seed(0)
    x = torch.randn(2, 6, 16)
    out = VisionTransformerEncoderLayer(d_model=16, nhead=4)(x)

    assert out.shape == x.shape
    assert torch.all(torch.isfinite(out))


def test_vit_encoder_layer_output_attentions():
    """`output_attentions=True` returns weights shaped `(n, s, s)`."""
    torch.manual_seed(0)
    x = torch.randn(2, 6, 16)
    out, attn = VisionTransformerEncoderLayer(d_model=16, nhead=4)(
        x, output_attentions=True)

    assert out.shape == x.shape
    assert attn.shape == (2, 6, 6)


@pytest.mark.parametrize("norm", ["layer", "rms"])
def test_vit_encoder_layer_norm_variants(norm):
    """Both `norm` variants run and preserve shape."""
    torch.manual_seed(0)
    x = torch.randn(2, 6, 16)
    out = VisionTransformerEncoderLayer(d_model=16, nhead=4, norm=norm)(x)

    assert out.shape == x.shape


@pytest.mark.parametrize("activation", ["gelu", "swiglu"])
def test_vit_encoder_layer_activation_variants(activation):
    """Both `activation` variants run and preserve shape."""
    torch.manual_seed(0)
    x = torch.randn(2, 6, 16)
    layer = VisionTransformerEncoderLayer(
        d_model=16, nhead=4, activation=activation)
    out = layer(x)

    assert out.shape == x.shape


def test_vit_encoder_layer_invalid_norm_raises():
    """An unrecognized `norm` value is rejected.

    The import-time type checker rejects it via the `Literal` hint; the
    source's own `ValueError` is the fallback when type checking is disabled.
    """
    with pytest.raises((ValueError, TypeCheckError)):
        VisionTransformerEncoderLayer(d_model=16, norm="invalid")  # type: ignore


def test_vit_encoder_layer_invalid_activation_raises():
    """An unrecognized `activation` value is rejected.

    The import-time type checker rejects it via the `Literal` hint; the
    source's own `ValueError` is the fallback when type checking is disabled.
    """
    with pytest.raises((ValueError, TypeCheckError)):
        VisionTransformerEncoderLayer(d_model=16, activation="invalid")  # type: ignore


def test_vit_encoder_layer_no_layer_scale():
    """`layer_scale_init_value <= 0` disables the learnable gammas."""
    layer = VisionTransformerEncoderLayer(
        d_model=16, layer_scale_init_value=0.0)
    assert layer.gamma1 is None
    assert layer.gamma2 is None


# AATransformerEncoder


def test_aa_transformer_encoder_preserves_shape():
    """The encoder preserves `(n, t, s, c)` shape."""
    torch.manual_seed(0)
    x = torch.randn(2, 3, 5, 16)
    out = AATransformerEncoder(d_model=16, nhead=4, num_layers=4)(x)

    assert out.shape == x.shape
    assert torch.all(torch.isfinite(out))


def test_aa_transformer_encoder_single_frame():
    """`t=1` (degenerate to a standard transformer) still runs correctly."""
    torch.manual_seed(0)
    x = torch.randn(2, 1, 5, 16)
    out = AATransformerEncoder(d_model=16, nhead=4, num_layers=4)(x)

    assert out.shape == x.shape


def test_aa_transformer_encoder_odd_layers_raises():
    """An odd `num_layers` is rejected due to the alternating pattern."""
    with pytest.raises(ValueError):
        AATransformerEncoder(d_model=16, num_layers=3)


def test_aa_transformer_encoder_output_attentions():
    """`output_attentions=True` returns frame-wise weights `(n, t, s, s)`."""
    torch.manual_seed(0)
    x = torch.randn(2, 3, 5, 16)
    out, attn = AATransformerEncoder(d_model=16, nhead=4, num_layers=4)(
        x, output_attentions=True)

    assert out.shape == x.shape
    assert attn.shape == (2, 3, 5, 5)
