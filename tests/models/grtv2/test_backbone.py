"""Tests for GRTv2 tokenizer, encoder, decoder, and end-to-end model."""

import functools

import pytest
import torch
from abstract_dataloader.ext import types
from jaxtyping import Float
from torch import Tensor, nn

from nrdk.models.grtv2 import (
    Decoder,
    Encoder,
    GRTv2,
    InputPositionalEncoder,
    LinearHead,
    OutputPositionalEncoder,
    Query,
    Tokenizer,
)
from nrdk.modules import AATransformerEncoder, ConvNextLayer

D_MODEL = 16
# (n, t, doppler, tx, rx, rng, ch)
SHAPE = (2, 3, 8, 2, 2, 16, 2)
D_RAW = SHAPE[3] * SHAPE[4] * SHAPE[6]
# Tokenizer with `stages=[1, 1]` and `downsample=[1, 2]` halves doppler/range.
D_TOK, R_TOK = SHAPE[2] // 2, SHAPE[5] // 2


@types.dataclass
class _Spectrum:
    """Minimal container satisfying the `nrdk.types.SpectrumData` protocol."""

    spectrum: Float[Tensor, "batch t doppler x1 x2 rng ch"]
    timestamps: Float[Tensor, "batch t"]


def _spectrum(seed=0):
    torch.manual_seed(seed)
    return torch.randn(*SHAPE)


def _timestamps():
    return torch.arange(SHAPE[1], dtype=torch.float32)[None].repeat(
        SHAPE[0], 1) * 0.1


def _tokenizer(mask_token=True):
    return Tokenizer(
        d_raw=D_RAW, d_tok=8, d_model=D_MODEL, stages=[1, 1],
        downsample=[1, 2], mask_token=mask_token,
        layer=functools.partial(ConvNextLayer, padding_mode="circular"))


def _encoder(n_registers=1, mask_token=True):
    return Encoder(
        tokenizer=_tokenizer(mask_token=mask_token),
        positional_encoder=InputPositionalEncoder(
            d_model=D_MODEL, d_features=8),
        stem=AATransformerEncoder(
            d_model=D_MODEL, nhead=2, dim_feedforward=32, num_layers=2,
            norm="rms", layer_scale_init_value=0.0),
        d_model=D_MODEL, n_registers=n_registers)


def _decoder():
    return Decoder(
        decoder=nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=D_MODEL, nhead=2, dim_feedforward=32,
                batch_first=True, norm_first=True),
            num_layers=1),
        positional_encoder=OutputPositionalEncoder(
            d_model=D_MODEL, d_features=8))


def _query(n=SHAPE[0], q=6, **overrides):
    torch.manual_seed(1)
    fields = {k: torch.randn(n, q) for k in ("t", "rng", "az", "el")}
    fields.update(overrides)
    return Query(**fields)


# Tokenizer


def test_tokenizer_output_shape():
    """Doppler and range are downsampled; time is preserved."""
    out = _tokenizer()(_spectrum())
    assert out.shape == (SHAPE[0], SHAPE[1], D_TOK, R_TOK, D_MODEL)


def test_tokenizer_mask_changes_output():
    """Masked positions are replaced before the convolutional encoder."""
    tokenizer = _tokenizer()
    x = _spectrum()
    mask = torch.zeros(SHAPE[0], SHAPE[1], SHAPE[2], SHAPE[5], dtype=torch.bool)
    mask[:, :, :SHAPE[2] // 2] = True

    assert not torch.allclose(tokenizer(x), tokenizer(x, mask=mask))


def test_tokenizer_without_mask_token():
    """Without a mask token, masked inputs are zero-filled."""
    tokenizer = _tokenizer(mask_token=False)
    assert tokenizer.mask_token is None
    mask = torch.ones(SHAPE[0], SHAPE[1], SHAPE[2], SHAPE[5], dtype=torch.bool)
    # Fully masked inputs are independent of the input values.
    a = tokenizer(_spectrum(seed=0), mask=mask)
    b = tokenizer(_spectrum(seed=1), mask=mask)
    assert torch.allclose(a, b)


# Encoder


@pytest.mark.parametrize("n_registers", [1, 2])
def test_encoder_output_shape(n_registers):
    """Output is `(n, t, tokens + registers, c)`."""
    out = _encoder(n_registers=n_registers)(_spectrum(), t=_timestamps())
    assert out.shape == (
        SHAPE[0], SHAPE[1], D_TOK * R_TOK + n_registers, D_MODEL)


def test_encoder_strip_registers():
    """`strip_registers` removes the register tokens."""
    out = _encoder(n_registers=2)(
        _spectrum(), t=_timestamps(), strip_registers=True)
    assert out.shape == (SHAPE[0], SHAPE[1], D_TOK * R_TOK, D_MODEL)


def test_encoder_mask_drops_tokens():
    """Fully-masked tokens are dropped from the sequence."""
    mask = torch.zeros(SHAPE[0], SHAPE[1], SHAPE[2], SHAPE[5], dtype=torch.bool)
    mask[:, :, :SHAPE[2] // 2] = True  # half of the doppler token rows
    out = _encoder()(_spectrum(), t=_timestamps(), mask=mask)
    assert out.shape == (SHAPE[0], SHAPE[1], D_TOK * R_TOK // 2 + 1, D_MODEL)


def test_encoder_partial_mask_keeps_tokens():
    """A token is only dropped if every contributing position is masked."""
    mask = torch.zeros(SHAPE[0], SHAPE[1], SHAPE[2], SHAPE[5], dtype=torch.bool)
    mask[:, :, ::2] = True  # every token is only partially masked
    out = _encoder()(_spectrum(), t=_timestamps(), mask=mask)
    assert out.shape == (SHAPE[0], SHAPE[1], D_TOK * R_TOK + 1, D_MODEL)


def test_encoder_spectrum_data_matches_tensor():
    """`SpectrumData` inputs use their own timestamps."""
    encoder = _encoder()
    encoder.eval()
    x, t = _spectrum(), _timestamps()
    assert torch.allclose(
        encoder(x, t=t), encoder(_Spectrum(spectrum=x, timestamps=t)))


def test_encoder_tensor_without_timestamps_raises():
    """Raw tensor inputs require explicit timestamps."""
    with pytest.raises(ValueError):
        _encoder()(_spectrum())


# Decoder


def test_decoder_output_shape():
    """One output token is produced per query."""
    latents = torch.randn(SHAPE[0], SHAPE[1], 5, D_MODEL)
    out = _decoder()(latents, _query(q=7))
    assert out.shape == (SHAPE[0], 7, D_MODEL)


# End-to-end


def test_grtv2_end_to_end():
    """A full GRTv2 model produces the configured output grid."""
    model = GRTv2(
        encoder=_encoder(), decoder=_decoder(),
        heads={"occ": LinearHead(
            d_model=D_MODEL, out_channels=1, output_size=[1, 8, 4],
            patch_size=[1, 4, 2], squeeze=True)},
        key="spectrum", queries={"occ": "occ_query"})
    x, t = _spectrum(), _timestamps()
    out = model({
        "spectrum": _Spectrum(spectrum=x, timestamps=t),
        "occ_query": _query(q=4)})
    assert out["occ"].shape == (SHAPE[0], 1, 8, 4)
    assert torch.all(torch.isfinite(out["occ"]))
