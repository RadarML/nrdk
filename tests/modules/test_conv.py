"""Tests for `nrdk.modules.conv`."""

import torch

from nrdk.modules.conv import ConvNextLayer


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
