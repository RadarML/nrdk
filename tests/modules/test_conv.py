"""Tests for `nrdk.modules.conv`."""

import pytest
import torch

from nrdk.modules.conv import (
    ConvDecoder,
    ConvDown,
    ConvEncoder,
    ConvNextLayer,
    ConvResidual,
    ConvUp,
)


def _small_input(n=2, c=8, h=6, w=6, seed=0):
    """Generate a small random `(n, c, h, w)` tensor."""
    torch.manual_seed(seed)
    return torch.randn(n, c, h, w)


def test_convnext_layer_preserves_shape():
    """A ConvNext residual block preserves input shape."""
    x = _small_input(c=8)
    out = ConvNextLayer(channels=8)(x)

    assert out.shape == x.shape
    assert torch.all(torch.isfinite(out))


def test_convnext_layer_scale_disabled_when_nonpositive():
    """`layer_scale_init_value <= 0` disables the learnable gamma entirely."""
    layer = ConvNextLayer(channels=8, layer_scale_init_value=0.0)
    assert layer.gamma is None

    out = layer(_small_input(c=8))
    assert torch.all(torch.isfinite(out))


def test_convnext_layer_scale_negative_also_disables_gamma():
    """A negative `layer_scale_init_value` also disables gamma."""
    layer = ConvNextLayer(channels=8, layer_scale_init_value=-1.0)
    assert layer.gamma is None


def test_convnext_layer_scale_initializes_gamma_parameter():
    """A positive `layer_scale_init_value` creates a per-channel gamma."""
    layer = ConvNextLayer(channels=8, layer_scale_init_value=0.5)

    assert layer.gamma is not None
    assert layer.gamma.shape == (8,)
    assert torch.allclose(layer.gamma, torch.full((8,), 0.5))


def test_convnext_layer_scale_proportional_to_init_value():
    """The residual contribution scales linearly with `gamma`.

    With identical weights (same seed), doubling `layer_scale_init_value`
    should double the residual `out - x`, since `gamma` uniformly scales the
    unscaled residual branch.
    """
    x = _small_input(c=8, seed=42)

    torch.manual_seed(7)
    layer_a = ConvNextLayer(channels=8, layer_scale_init_value=0.5)
    torch.manual_seed(7)
    layer_b = ConvNextLayer(channels=8, layer_scale_init_value=1.5)

    residual_a = layer_a(x) - x
    residual_b = layer_b(x) - x

    assert torch.allclose(residual_b, 3.0 * residual_a, atol=1e-4)


def test_convnext_layer_padding_mode_preserves_shape():
    """Non-default padding modes still preserve shape."""
    x = _small_input(c=8)
    out = ConvNextLayer(channels=8, padding_mode="reflect")(x)
    assert out.shape == x.shape


def test_convnext_layer_expansion_ratio_does_not_change_output_shape():
    """The inverted bottleneck expansion ratio is internal to the block."""
    x = _small_input(c=8)
    out = ConvNextLayer(channels=8, expansion_ratio=2.0)(x)
    assert out.shape == x.shape


# ConvResidual


def test_conv_residual_preserves_shape():
    """A residual block preserves input shape."""
    x = _small_input(c=8)
    out = ConvResidual(dim=8)(x)

    assert out.shape == x.shape
    assert torch.all(torch.isfinite(out))


def test_conv_residual_no_layer_scale():
    """`layer_scale_init_value <= 0` disables the learnable gamma."""
    block = ConvResidual(dim=8, layer_scale_init_value=0.0)
    assert block.gamma is None

    out = block(_small_input(c=8))
    assert torch.all(torch.isfinite(out))


def test_conv_residual_padding_mode():
    """Non-default padding modes run without error."""
    x = _small_input(c=8)
    out = ConvResidual(dim=8, padding_mode="zeros")(x)

    assert out.shape == x.shape


# ConvDown / ConvUp


def test_conv_down_shape():
    """Downsampling divides spatial dims and remaps channels."""
    x = _small_input(c=4, h=8, w=8)
    out = ConvDown(d_in=4, d_out=16, downsample=2, depth=1)(x)

    assert out.shape == (2, 16, 4, 4)


def test_conv_down_asymmetric_downsample():
    """A `(height, width)` downsample tuple applies per-axis factors."""
    x = _small_input(c=4, h=8, w=16)
    out = ConvDown(d_in=4, d_out=8, downsample=(2, 4), depth=1)(x)

    assert out.shape == (2, 8, 4, 4)


def test_conv_up_shape():
    """Upsampling multiplies spatial dims and remaps channels."""
    x = _small_input(c=16, h=4, w=4)
    out = ConvUp(d_in=16, d_out=4, upsample=2, depth=1)(x)

    assert out.shape == (2, 4, 8, 8)


def test_conv_down_up_roundtrip_shape():
    """A down stage followed by a matching up stage restores shape."""
    x = _small_input(c=4, h=8, w=8)
    down = ConvDown(d_in=4, d_out=16, downsample=2, depth=1)
    up = ConvUp(d_in=16, d_out=4, upsample=2, depth=1)

    out = up(down(x))
    assert out.shape == x.shape


# ConvEncoder / ConvDecoder


def test_conv_encoder_default_shape_and_latent_dim():
    """Default encoder downsamples spatially and grows channel width."""
    x = _small_input(c=24, h=16, w=16)
    encoder = ConvEncoder(stages=(1, 1, 1), d_in=24, width=8)

    out = encoder(x)

    assert encoder.latent_dim == 8 * 2**2
    assert out.shape == (2, encoder.latent_dim, 4, 4)


def test_conv_encoder_d_out_projects_channels():
    """`d_out` adds a final 1x1 conv projecting to the target channel dim."""
    x = _small_input(c=24, h=8, w=8)
    encoder = ConvEncoder(stages=(1, 1), d_in=24, width=8, d_out=5)

    out = encoder(x)
    assert out.shape[1] == 5


def test_conv_encoder_empty_stages_raises():
    """An empty `stages` sequence is rejected."""
    with pytest.raises(ValueError):
        ConvEncoder(stages=())


def test_conv_decoder_default_shape_and_latent_dim():
    """Default decoder upsamples spatially and shrinks channel width."""
    latent = _small_input(c=32, h=4, w=4)
    decoder = ConvDecoder(stages=(1, 1, 1), d_in=8, width=8)

    out = decoder(latent)

    assert decoder.latent_dim == 8 * 2**2
    assert out.shape == (2, 8, 16, 16)


def test_conv_decoder_empty_stages_raises():
    """An empty `stages` sequence is rejected."""
    with pytest.raises(ValueError):
        ConvDecoder(stages=())


def test_conv_encoder_decoder_roundtrip_shape():
    """An encoder followed by a matching decoder restores the input shape."""
    x = _small_input(c=24, h=16, w=16)
    encoder = ConvEncoder(stages=(1, 1, 1), d_in=24, width=8)
    decoder = ConvDecoder(stages=(1, 1, 1), d_in=24, width=8)

    out = decoder(encoder(x))
    assert out.shape == x.shape
