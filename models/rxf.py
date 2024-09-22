"""Radar Transformer."""

import torch
from beartype.typing import Literal, Sequence, cast
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn

from deepradar import modules


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
        patch: input (doppler, range) patch size.
    """

    def __init__(
        self, enc_layers: int = 5, dec_layers: list[int] = [5, 3, 1],
        dim: int = 768, ff_ratio: float = 4.0, heads: int = 12,
        dropout: float = 0.1, activation: str = 'GELU',
        patch: Sequence[int] = (16, 1, 1, 16)
    ) -> None:
        super().__init__()

        self.dec_layers = dec_layers

        self.patch = modules.PatchMerge(
            d_in=2, d_out=dim, scale=patch, norm=False)
        self.pos = modules.Sinusoid()
        self.readout = modules.Readout(d_model=dim)

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
        embedded = self.pos(self.patch(x))
        flat = rearrange(embedded, "n d r a e c -> n (d r a e) c")

        encoded = [self.readout(flat)]
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
        out_dim: output channels; if `=0`, the dimension is omitted entirely,
            i.e. `(h, w)` instead of `(h, w, c)`.
    """

    def __init__(
        self, key: str, dec_layers: int = 3, dim: int = 768,
        ff_ratio: float = 4.0, heads: int = 12, dropout: float = 0.1,
        activation: str = 'GELU', shape: Sequence[int] = (1024, 256),
        patch: Sequence[int] = (16, 16), out_dim: int = 0
    ) -> None:
        super().__init__()

        self.key = key
        self.out_dim = out_dim

        self.layers = nn.ModuleList([
            modules.TransformerDecoder(
                d_feedforward=int(ff_ratio * dim), d_model=dim, n_head=heads,
                dropout=dropout, activation=activation)
            for _ in range(dec_layers)])

        self.query = modules.BasisChange(
            shape=(shape[0] // patch[0], shape[1] // patch[1]))

        self.unpatch = modules.Unpatch2D(
            output_size=(shape[0], shape[1], max(1, self.out_dim)),
            features=dim, size=cast(tuple[int, int], tuple(patch)))

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
        if self.out_dim == 0:
            out = out[:, 0, :, :]

        return {self.key: out}


class VectorDecoder(nn.Module):
    """Generic MLP-based vector decoder without spatial dimensions.

    Always uses the first encoder output tensor, and supports the following
    reduction strategies for that tensor:

    - `last`: take the last spatial feature (nominally a readout token).
    - `max`, `avg`: max or average pooling over all spatial dimensions.

    Args:
        key: target key, e.g. `bev`, `depth`.
        layers: MLP architecture.
        dropout: dropout ratio during training.
        activation: activation function; specify as a name (i.e. corresponding
            to a class in `torch.nn`).
        dim: input features.
        out_dim: output features.
        reduce: reduction strategy.
        channels_first: whether input channels are in channels-spatial (`NCHW`)
            order instead of spatial-channels order (`NHWC`).
    """

    def __init__(
        self, key: str, layers: list[int] = [512, 512], dropout: float = 0.1,
        activation: str = 'GELU', dim: int = 768, out_dim: int = 3,
        strategy: Literal["last", "maxpool", "avgpool"] = "last",
        channels_first: bool = False
    ) -> None:
        super().__init__()

        self.key = key
        self.strategy = strategy
        self.channels_first = channels_first

        _layers = []
        for d1, d2 in zip(([dim] + layers)[:-1], layers):
            _layers += [
                nn.Linear(d1, d2, bias=True),
                getattr(nn, activation)(),
                nn.Dropout(dropout)]

        _layers.append(nn.Linear(([dim] + layers)[-1], out_dim))
        self.mlp = nn.Sequential(*_layers)

    def forward(
        self, encoded: list[Float[Tensor, "?n *?s ?c"]]
    ) -> dict[str, Float[Tensor, "n f"]]:
        """Apply decoder.

        Args:
            encoded: list of encoded values. Only the last token of the last
                tensor (nominally the readout token) is used.

        Returns:
            A tensor with the specified number of features.
        """
        if self.channels_first:
            n, c, *s = encoded[0].shape
            x = encoded[0].reshape(n, c, -1).permute(0, 2, 1)
        else:
            n, *s, c = encoded[0].shape
            x = encoded[0].reshape(n, -1, c)

        if self.strategy == "last":
            x = x[:, -1]
        elif self.strategy == "maxpool":
            x = torch.max(x, dim=1).values
        elif self.strategy == "avgpool":
            x = torch.mean(x, dim=1)
        else:
            raise ValueError(f"Invalid strategy: {self.strategy}")

        return {self.key: self.mlp(x)}
