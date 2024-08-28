"""Radar Transformer."""

import torch
from torch import Tensor, nn
from einops import rearrange

from beartype.typing import Union, cast
from jaxtyping import Float

from deepradar import modules


class BasisChange(nn.Module):
    """Create "change-of-basis" query.

    Uses a 'reference vector', e.g. the output for a readout token or the
    token-wise mean of the output.

    Args:
        shape: query shape.
    """

    def __init__(
        self, shape: Union[list[int], tuple[int, ...]] = (16, 16)
    ) -> None:
        super().__init__()

        self.pos = modules.Sinusoid()
        self.shape = shape

    def forward(self, x: Float[Tensor, "n c"]) -> Float[Tensor, "n t2 c"]:

        idxs = [slice(None)] + [None] * len(self.shape) + [slice(None)]
        query = self.pos(
            torch.tile(x[idxs], (1, *self.shape, 1))
        ).reshape(x.shape[0], -1, x.shape[-1])

        return query


Shape2 = Union[list[int], tuple[int, int]]
"""List of 2 ints (i.e. loaded from json), or tuple of ints."""


def assert_shape2(x: Shape2) -> tuple[int, int]:
    """Type validation of Shape2 and casting as tuple."""
    return cast(tuple[int, int], tuple(x))


class TransformerEncoder(nn.Module):
    """Radar Doppler Transformer.

    Only a selected subset of hidden layers are passed to the decoder, and
    are specified by `dec_layers` as follows:

    - Each entry in the list indicates the index of the encoder layer output
      that will be passed to the decoder. The length must match `dec_layers` in
      the decoder.
    - Indexing starts from 0 at the output of the patch projection (e.g. the
      output of the first encoder layer is index 1).

    Args:
        enc_layers: number of encoder layers.
        dec_layers: encoded layer indices to pass to the decoder.
        dim: hidden dimension.
        ff_ratio: expansion ratio for feedforward blocks.
        heads: number of heads for multiheaded attention.
        dropout: dropout ratio during training.
        activation: activation function; specify as a name (i.e. corresponding
            to a class in `torch.nn`).
        patch: input range-doppler patch size.
    """

    def __init__(
        self, enc_layers: int = 5, dec_layers: list[int] = [5, 3, 1],
        dim: int = 768, ff_ratio: float = 4.0, heads: int = 12,
        dropout: float = 0.1, activation: str = 'GELU',
        patch: Shape2 = (16, 16)
    ) -> None:
        super().__init__()

        patch = assert_shape2(patch)
        self.dec_layers = dec_layers

        self.patch = modules.Patch4D(channels=2, features=dim, size=patch)
        self.pos = modules.Sinusoid()

        self.readout = nn.Parameter(data=torch.normal(0, 0.02, (dim,)))

        self.layers = nn.ModuleList([
            modules.TransformerLayer(
                d_feedforward=int(ff_ratio * dim), d_model=dim, n_head=heads,
                dropout=dropout, activation=activation)
            for _ in range(enc_layers)])

    def forward(
        self, x: Float[Tensor, "n d a e r c"]
    ) -> list[Float[Tensor, "n s c"]]:
        """Apply radar transformer.

        Args:
            x: input batch, with batch-doppler-azimuth-elevation-range-iq axis
                order.

        Returns:
            Encoding output.
        """
        patch = self.patch(rearrange(x, "n d a e r c -> n c d r a e"))
        embedded = self.pos(patch)

        x0 = rearrange(embedded, "n d r a e c -> n (d r a e) c")
        readout = torch.tile(self.readout[None, None, :], (x0.shape[0], 1, 1))
        # The output type of `rearrange` isn't inferred correctly.
        x0 = torch.concatenate([x0, readout], axis=1)  # type: ignore

        encoded = [x0]
        for layer in self.layers:
            encoded.append(layer(encoded[-1]))

        return [encoded[i] for i in self.dec_layers]


class Transformer2DDecoder(nn.Module):
    """Radar transformer 2D tensor decoder.

    Args:
        key: target key, e.g. `bev`, `depth`.
        dec_layers: number of decoder layers.
        dim: hidden dimension; should be the same as the encoder.
        ff_ratio: expansion ratio for feedforward blocks.
        heads: number of attention heads.
        dropout: dropout during training.
        activation: activation function to use.
        shape: output shape; should be a 2 element list or tuple.
        patch: patch size to use for unpatching. Must evenly divide `shape`.
        out_dim: output channels; if `=1`, the dimension is omitted entirely,
            i.e. `(h, w)` instead of `(h, w, c)`.
    """

    def __init__(
        self, key: str, dec_layers: int = 3, dim: int = 768,
        ff_ratio: float = 4.0, heads: int = 12, dropout: float = 0.1,
        activation: str = 'GELU',
        shape: Shape2 = (1024, 256), patch: Shape2 = (16, 16), out_dim: int = 1
    ) -> None:
        super().__init__()

        self.key = key
        self.out_dim = out_dim
        shape = assert_shape2(shape)
        patch = assert_shape2(patch)

        self.layers = nn.ModuleList([
            modules.TransformerDecoder(
                d_feedforward=int(ff_ratio * dim), d_model=dim, n_head=heads,
                dropout=dropout, activation=activation)
            for _ in range(dec_layers)])

        self.query = BasisChange(
            shape=(shape[0] // patch[0], shape[1] // patch[1]))

        self.unpatch = modules.Unpatch2D(
            output_size=(shape[0], shape[1], self.out_dim),
            features=dim, size=patch)

    def forward(
        self, encoded: list[Float[Tensor, "n s c"]]
    ) -> dict[str, Float[Tensor, "n h w ..."]]:
        """Apply decoder.

        Args:
            encoded: list of encoded values. Each tensor should be the same
                size, and use batch-spatial-channel order. The last spatial
                element of each tensor should correspond to a readout token.

        Returns:
            2-dimensional output; only a single key (e.g. the specified `key`)
            is decoded.
        """
        out = self.query(encoded[0][:, -1, :])
        for enc, layer in zip(encoded, self.layers):
            out = layer(out, enc)

        out = self.unpatch(out)
        if self.out_dim == 1:
            out = out[:, 0, :, :]

        return {self.key: out}


class TransformerFixedDecoder(nn.Module):
    """Radar transformer MLP decoder without spatial dimensions.

    Args:
        key: target key, e.g. `bev`, `depth`.
        layers: MLP architecture.
        dropout: dropout ratio during training.
        activation: activation function; specify as a name (i.e. corresponding
            to a class in `torch.nn`).
        dim: input features.
        out_dim: output features.
    """

    def __init__(
        self, key: str, layers: list[int] = [512, 512], dropout: float = 0.1,
        activation: str = 'GELU', dim: int = 768, out_dim: int = 3
    ) -> None:
        _layers = []
        for d1, d2 in zip(([dim] + layers)[:-1], layers):
            _layers += [
                nn.Linear(d1, d2, bias=True) + getattr(nn, activation)()
                + nn.Dropout(dropout)]

        _layers.append(nn.Linear(([dim] + layers)[-1], out_dim))
        self.mlp = nn.Sequential(*_layers)

    def forward(
        self, encoded: list[Float[Tensor, "n s c"]]
    ) -> dict[str, Float[Tensor, "n f"]]:
        """Apply decoder.

        Args:
            encoded: list of encoded values. Only the last token of the last
                tensor (nominally the readout token) is used.

        Returns:
            A tensor with the specified number of features.
        """
        readout = encoded[-1][:, -1]
        return {self.key: self.mlp(readout)}
