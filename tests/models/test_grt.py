"""Tests for GRT models."""

from typing import Literal

import pytest
import torch
from torch import nn

from nrdk.models.grt import MLPVectorDecoder, TransformerTensorDecoder


def _decoder_layer(d_model=16, nhead=2, dim_feedforward=32):
    """Build a batch-first transformer decoder layer."""
    return nn.TransformerDecoderLayer(
        d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
        batch_first=True)


def _encoded(n=3, mem_len=5, d_model=16, seed=0):
    """Build a small `(n, mem_len + 1, d_model)` encoded tensor.

    The last token along the sequence axis is the readout token used to seed
    the query; the rest is treated as decoder memory.
    """
    torch.manual_seed(seed)
    return torch.randn(n, mem_len + 1, d_model)


# TransformerTensorDecoder


def test_transformer_tensor_decoder_out_dim_zero_shape():
    """`out_dim=0` drops the trailing channel axis entirely."""
    d_model = 16
    decoder = TransformerTensorDecoder(
        _decoder_layer(d_model=d_model), d_model=d_model, num_layers=1,
        shape=(8, 8), patch=(4, 4), out_dim=0)

    out = decoder(_encoded(d_model=d_model))
    assert out.shape == (3, 8, 8)


@pytest.mark.parametrize("out_dim", [1, 3])
def test_transformer_tensor_decoder_out_dim_positive_shape(out_dim):
    """`out_dim > 0` keeps a trailing channel axis of that size."""
    d_model = 16
    decoder = TransformerTensorDecoder(
        _decoder_layer(d_model=d_model), d_model=d_model, num_layers=1,
        shape=(8, 8), patch=(4, 4), out_dim=out_dim)

    out = decoder(_encoded(d_model=d_model))
    assert out.shape == (3, 8, 8, out_dim)


def test_transformer_tensor_decoder_finite_output():
    """Output values are finite."""
    d_model = 16
    decoder = TransformerTensorDecoder(
        _decoder_layer(d_model=d_model), d_model=d_model, num_layers=2,
        shape=(8, 16), patch=(4, 8), out_dim=2, scale=(1.0, 2.0), w_min=0.5)

    out = decoder(_encoded(d_model=d_model, mem_len=7))
    assert out.shape == (3, 8, 16, 2)
    assert torch.all(torch.isfinite(out))


def _mvd(
    layers=[16], dropout=0.0, activation="relu", dim=4, shape=[2],
    strategy: Literal["last", "maxpool", "avgpool"] = "last",
    channels_first=False
):
    return MLPVectorDecoder(
        layers=layers, dropout=dropout, activation=activation, dim=dim,
        shape=shape, strategy=strategy, channels_first=channels_first)


@pytest.mark.parametrize("strategy", ["last", "maxpool", "avgpool"])
def test_mlp_vector_decoder_strategy_matches_manual_pooling(strategy):
    """Each strategy pools the flattened spatial axis as documented."""
    torch.manual_seed(0)
    model = _mvd(strategy=strategy)
    model.eval()
    x = torch.randn(3, 5, 4)  # n s c

    out = model(x)

    if strategy == "last":
        pooled = x[:, -1]
    elif strategy == "maxpool":
        pooled = torch.max(x, dim=1).values
    else:
        pooled = torch.mean(x, dim=1)
    expected = model.mlp(pooled).reshape(x.shape[0], 2)

    assert out.shape == (3, 2)
    assert torch.allclose(out, expected)


def test_mlp_vector_decoder_invalid_strategy_raises():
    """An invalid strategy raises `ValueError` from `forward`."""
    model = _mvd(strategy="last")
    # `strategy` is a `Literal` checked by beartype at construction time, so
    # an invalid value must be assigned directly to reach the source's own
    # `raise ValueError` branch in `forward`.
    model.strategy = "bogus"

    with pytest.raises(ValueError):
        model(torch.randn(2, 5, 4))


def test_mlp_vector_decoder_channels_first_matches_channels_last():
    """`channels_first` reorders axes but computes the same reduction."""
    torch.manual_seed(1)
    x_first = torch.randn(2, 4, 6)  # n c s
    x_last = x_first.permute(0, 2, 1).contiguous()  # n s c

    model_first = _mvd(strategy="avgpool", channels_first=True)
    model_last = _mvd(strategy="avgpool", channels_first=False)
    model_last.load_state_dict(model_first.state_dict())
    model_first.eval()
    model_last.eval()

    out_first = model_first(x_first)
    out_last = model_last(x_last)
    assert torch.allclose(out_first, out_last, atol=1e-5)


def test_mlp_vector_decoder_output_shape_matches_configured_shape():
    """The output is reshaped to the configured (single-axis) `shape`."""
    model = _mvd(shape=[7], strategy="maxpool")
    out = model(torch.randn(2, 5, 4))
    assert out.shape == (2, 7)
