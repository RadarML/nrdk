"""Tests for nrdk.framework.architectures."""

import pytest
import torch
from torch import Tensor, nn

from nrdk.framework.architectures import Output, TokenizerEncoderDecoder


class _RecordingPassthrough(nn.Module):
    """Module stub that records its input and returns a scaled tensor."""

    def __init__(self, scale: float = 1.0) -> None:
        super().__init__()
        self.scale = scale
        self.calls: list[Tensor] = []

    def forward(self, x: Tensor) -> Tensor:
        self.calls.append(x)
        return x * self.scale


class _OutputReturning(nn.Module):
    """Module stub that returns an `Output` side-channel value."""

    def __init__(self, reg_key: str, scale: float = 1.0) -> None:
        super().__init__()
        self.reg_key = reg_key
        self.scale = scale
        self.calls: list[Tensor] = []

    def forward(self, x: Tensor) -> Output:
        self.calls.append(x)
        return Output(stage=x * self.scale, output={self.reg_key: x.sum()})


class _RecordingDecoder(nn.Module):
    """Decoder stub that records its input and returns a scaled tensor."""

    def __init__(self, scale: float = 1.0) -> None:
        super().__init__()
        self.scale = scale
        self.calls: list[Tensor] = []

    def forward(self, x: Tensor) -> Tensor:
        self.calls.append(x)
        return x * self.scale


class _UnsqueezingDecoder(nn.Module):
    """Decoder stub that appends two singleton trailing axes."""

    def forward(self, x: Tensor) -> Tensor:
        return x.view(*x.shape, 1, 1)


def test_no_decoder_raises():
    """At least one decoder is required."""
    with pytest.raises(ValueError):
        TokenizerEncoderDecoder(decoder={})


def test_missing_input_key_raises():
    """A data dict missing the configured `key` raises `KeyError`."""
    model = TokenizerEncoderDecoder(
        decoder={"out": _RecordingDecoder()}, key="spectrum")
    with pytest.raises(KeyError):
        model({"not_spectrum": torch.randn(2, 4)})


def test_tokenizer_none_passes_input_directly_to_encoder():
    """With `tokenizer=None`, the encoder receives the raw input key."""
    x = torch.randn(2, 4)
    encoder = _RecordingPassthrough(scale=1.0)
    model = TokenizerEncoderDecoder(
        tokenizer=None, encoder=encoder, decoder={"out": _RecordingDecoder()})
    model({"spectrum": x})

    assert len(encoder.calls) == 1
    assert torch.equal(encoder.calls[0], x)


def test_encoder_none_passes_tokens_directly_to_decoder():
    """With `encoder=None`, the decoder receives the tokenizer output."""
    x = torch.randn(2, 4)
    tokenizer = _RecordingPassthrough(scale=2.0)
    decoder = _RecordingDecoder(scale=1.0)
    model = TokenizerEncoderDecoder(
        tokenizer=tokenizer, encoder=None, decoder={"out": decoder})
    model({"spectrum": x})

    assert len(decoder.calls) == 1
    assert torch.allclose(decoder.calls[0], x * 2.0)


def test_tokenizer_and_encoder_none_passes_input_directly_to_decoder():
    """With both `tokenizer=None` and `encoder=None`, decoder sees raw input."""
    x = torch.randn(2, 4)
    decoder = _RecordingDecoder(scale=1.0)
    model = TokenizerEncoderDecoder(
        tokenizer=None, encoder=None, decoder={"out": decoder})
    model({"spectrum": x})

    assert torch.equal(decoder.calls[0], x)


def test_tokenizer_output_side_channel_merged():
    """A tokenizer returning `Output` merges `.output` into the result."""
    x = torch.randn(2, 4)
    tokenizer = _OutputReturning("reg_tok", scale=3.0)
    decoder = _RecordingDecoder(scale=1.0)
    model = TokenizerEncoderDecoder(
        tokenizer=tokenizer, encoder=None, decoder={"out": decoder},
        squeeze=False)

    result = model({"spectrum": x})

    assert "reg_tok" in result
    assert torch.allclose(result["reg_tok"], x.sum())
    # `.stage` (not the `Output` wrapper) is what flows downstream.
    assert torch.allclose(decoder.calls[0], x * 3.0)
    assert torch.allclose(result["out"], x * 3.0)


def test_encoder_output_side_channel_merged():
    """An encoder returning `Output` merges `.output` into the result."""
    x = torch.randn(2, 4)
    encoder = _OutputReturning("reg_enc", scale=5.0)
    decoder = _RecordingDecoder(scale=1.0)
    model = TokenizerEncoderDecoder(
        tokenizer=None, encoder=encoder, decoder={"out": decoder},
        squeeze=False)

    result = model({"spectrum": x})

    assert "reg_enc" in result
    assert torch.allclose(result["reg_enc"], x.sum())
    assert torch.allclose(decoder.calls[0], x * 5.0)


def test_tokenizer_and_encoder_output_side_channels_both_merged():
    """Side channels from both tokenizer and encoder are merged together."""
    x = torch.randn(2, 4)
    tokenizer = _OutputReturning("reg_tok", scale=2.0)
    encoder = _OutputReturning("reg_enc", scale=3.0)
    decoder = _RecordingDecoder(scale=1.0)
    model = TokenizerEncoderDecoder(
        tokenizer=tokenizer, encoder=encoder, decoder={"out": decoder},
        squeeze=False)

    result = model({"spectrum": x})

    assert set(result.keys()) == {"reg_tok", "reg_enc", "out"}
    # Encoder receives tokenizer's `.stage` output.
    assert torch.allclose(encoder.calls[0], x * 2.0)
    # Decoder receives encoder's `.stage` output.
    assert torch.allclose(decoder.calls[0], x * 2.0 * 3.0)


def test_multiple_decoders_produce_multiple_output_keys():
    """Each decoder in the mapping produces its own output key."""
    x = torch.randn(2, 4)
    decoders = {
        "a": _RecordingDecoder(scale=2.0),
        "b": _RecordingDecoder(scale=10.0),
    }
    model = TokenizerEncoderDecoder(decoder=decoders, squeeze=False)
    result = model({"spectrum": x})

    assert set(result.keys()) == {"a", "b"}
    assert torch.allclose(result["a"], x * 2.0)
    assert torch.allclose(result["b"], x * 10.0)


def test_squeeze_true_removes_trailing_singleton_dims():
    """`squeeze=True` removes non-batch, non-temporal singleton axes."""
    x = torch.randn(2, 3)
    model = TokenizerEncoderDecoder(
        decoder={"out": _UnsqueezingDecoder()}, squeeze=True)
    result = model({"spectrum": x})

    assert result["out"].shape == (2, 3)
    assert torch.allclose(result["out"], x)


def test_squeeze_false_keeps_trailing_singleton_dims():
    """`squeeze=False` leaves the decoder output shape untouched."""
    x = torch.randn(2, 3)
    model = TokenizerEncoderDecoder(
        decoder={"out": _UnsqueezingDecoder()}, squeeze=False)
    result = model({"spectrum": x})

    assert result["out"].shape == (2, 3, 1, 1)


def test_custom_input_key():
    """A non-default `key` is used to select the model input."""
    x = torch.randn(2, 4)
    decoder = _RecordingDecoder(scale=1.0)
    model = TokenizerEncoderDecoder(
        decoder={"out": decoder}, key="custom_key")
    model({"custom_key": x})

    assert torch.equal(decoder.calls[0], x)
