"""Tests for the `GRTv2` model wrapper."""

import pytest
import torch
from torch import Tensor, nn

from nrdk.models.grtv2 import GRTv2


class _RecordingEncoder(nn.Module):
    """Encoder stub that records its call arguments."""

    def __init__(self):
        super().__init__()
        self.calls: list[tuple[Tensor, Tensor | None]] = []

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        self.calls.append((x, mask))
        return x


class _AddDecoder(nn.Module):
    """Decoder stub combining latents and a query tensor."""

    def forward(self, latents: Tensor, query: Tensor) -> Tensor:
        return latents + query


class _ScaleHead(nn.Module):
    """Head stub returning a scaled tensor."""

    def __init__(self, factor: float = 1.0):
        super().__init__()
        self.factor = factor

    def forward(self, decoded: Tensor) -> Tensor:
        return decoded * self.factor


class _DictHead(nn.Module):
    """Head stub returning multiple named outputs."""

    def forward(self, decoded: Tensor) -> dict[str, Tensor]:
        return {"a": decoded, "b": decoded * 2}


def test_grtv2_mismatched_queries_heads_raises():
    """`queries` and `heads` with different key sets raise `ValueError`."""
    with pytest.raises(ValueError):
        GRTv2(
            encoder=_RecordingEncoder(), decoder=_AddDecoder(),
            heads={"h1": _ScaleHead()}, queries={"h2": "q"})


def test_grtv2_missing_input_key_raises():
    """A data dict missing the spectrum `key` raises `KeyError`."""
    model = GRTv2(
        encoder=_RecordingEncoder(), decoder=_AddDecoder(),
        heads={"h1": _ScaleHead()}, queries={"h1": "q1"})

    with pytest.raises(KeyError):
        model({"q1": torch.randn(2, 4)})


def test_grtv2_missing_mask_key_raises():
    """A configured `mask_key` missing from the data dict raises `KeyError`."""
    model = GRTv2(
        encoder=_RecordingEncoder(), decoder=_AddDecoder(),
        heads={"h1": _ScaleHead()}, queries={"h1": "q1"},
        mask_key="mask")

    with pytest.raises(KeyError):
        model({"spectrum": torch.randn(2, 4), "q1": torch.randn(2, 4)})


def test_grtv2_missing_query_key_raises():
    """A missing decoder query data key raises `KeyError`."""
    model = GRTv2(
        encoder=_RecordingEncoder(), decoder=_AddDecoder(),
        heads={"h1": _ScaleHead()}, queries={"h1": "q1"})

    with pytest.raises(KeyError):
        model({"spectrum": torch.randn(2, 4)})


def test_grtv2_forward_basic():
    """A tensor-returning head is stored under its head name."""
    torch.manual_seed(0)
    x = torch.randn(2, 4)
    query = torch.randn(2, 4)

    model = GRTv2(
        encoder=_RecordingEncoder(), decoder=_AddDecoder(),
        heads={"h1": _ScaleHead(factor=2.0)}, queries={"h1": "q1"})
    outputs = model({"spectrum": x, "q1": query})

    assert set(outputs.keys()) == {"h1"}
    assert torch.allclose(outputs["h1"], (x + query) * 2.0)


def test_grtv2_forward_dict_head():
    """A dict-returning head's keys are merged directly into the output."""
    torch.manual_seed(0)
    x = torch.randn(2, 4)
    query = torch.randn(2, 4)

    model = GRTv2(
        encoder=_RecordingEncoder(), decoder=_AddDecoder(),
        heads={"h1": _DictHead()}, queries={"h1": "q1"})
    outputs = model({"spectrum": x, "q1": query})

    assert set(outputs.keys()) == {"a", "b"}
    assert torch.allclose(outputs["a"], x + query)
    assert torch.allclose(outputs["b"], (x + query) * 2)


def test_grtv2_forward_multiple_heads():
    """Each head is decoded with its own query and stored separately."""
    torch.manual_seed(0)
    x = torch.randn(2, 4)
    q1 = torch.randn(2, 4)
    q2 = torch.randn(2, 4)

    model = GRTv2(
        encoder=_RecordingEncoder(), decoder=_AddDecoder(),
        heads={"h1": _ScaleHead(2.0), "h2": _ScaleHead(3.0)},
        queries={"h1": "q1", "h2": "q2"})
    outputs = model({"spectrum": x, "q1": q1, "q2": q2})

    assert torch.allclose(outputs["h1"], (x + q1) * 2.0)
    assert torch.allclose(outputs["h2"], (x + q2) * 3.0)


def test_grtv2_forward_passes_mask_to_encoder():
    """A configured `mask_key` is forwarded to the encoder."""
    torch.manual_seed(0)
    x = torch.randn(2, 4)
    query = torch.randn(2, 4)
    mask = torch.ones(2, 4, dtype=torch.bool)
    encoder = _RecordingEncoder()

    model = GRTv2(
        encoder=encoder, decoder=_AddDecoder(),
        heads={"h1": _ScaleHead()}, queries={"h1": "q1"},
        mask_key="mask")
    model({"spectrum": x, "q1": query, "mask": mask})

    assert len(encoder.calls) == 1
    called_x, called_mask = encoder.calls[0]
    assert torch.equal(called_x, x)
    assert called_mask is not None
    assert torch.equal(called_mask, mask)


def test_grtv2_forward_no_mask_key_passes_none():
    """Without a configured `mask_key`, the encoder receives `mask=None`."""
    torch.manual_seed(0)
    x = torch.randn(2, 4)
    query = torch.randn(2, 4)
    encoder = _RecordingEncoder()

    model = GRTv2(
        encoder=encoder, decoder=_AddDecoder(),
        heads={"h1": _ScaleHead()}, queries={"h1": "q1"})
    model({"spectrum": x, "q1": query})

    assert encoder.calls[0][1] is None
