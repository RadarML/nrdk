"""Radar Transformer."""

import lightning as L
import numpy as np
import torch
from beartype.typing import Literal, Optional, Sequence
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn

from deepradar import modules


class TransformerEncoder(L.LightningModule):
    """Radar Doppler Transformer.

    Only a selected subset of hidden layers are passed to the decoder, and
    are specified by `dec_layers` as follows:

    - Each entry in the list indicates the index of the encoder layer output
      that will be passed to the decoder. The length must match `dec_layers` in
      the decoder.
    - Indexing starts from 0 at the output of the patch projection (e.g. the
      output of the first encoder layer is index 1).

    Args:
        layers: number of encoder layers.
        dec_layers: which layers to pass to the decoder. If `None` (default),
            only the output of the last layer is returned.
        dim: hidden dimension.
        ff_ratio: expansion ratio for feedforward blocks.
        head_dim: number of dimensions per head for multihead attention.
        dropout: dropout ratio during training.
        activation: activation function; specify as a name (i.e. corresponding
            to a class in `torch.nn`).
        patch: input (doppler, azimuth, elevation, range) patch size.
        pos_scale: position embedding scale (i.e. the spatial range that this
            axis corresponds to). If `None`, only the global scale is used.
        global_scale: scalar constant to multiply scale by for convenience of
            representation; yields a net scale of `scale * global_scale`.
        input_channels: number of input channels.
        positions: type of positional embedding. `flat`: flattened positional
            embeddings, similar to the original ViT; `nd`: n-dimensional
            embeddings, splitting the input features into `d` equal chunks
            encoding each axis separately.
    """

    def __init__(
        self, layers: int = 5, dec_layers: Optional[Sequence[int]] = None,
        dim: int = 768, ff_ratio: float = 4.0, head_dim: int = 64,
        dropout: float = 0.1, activation: str = 'GELU',
        patch: Sequence[int] = (16, 1, 1, 16),
        pos_scale: Optional[Sequence[float]] = None, global_scale: float = 1.0,
        input_channels: int = 2,
        positions: Literal["flat", "nd"] = "flat",
    ) -> None:
        super().__init__()

        if len(patch) != 4:
            raise ValueError("Must specify a 4D patch size.")

        self.patch = modules.PatchMerge(
            d_in=input_channels, d_out=dim, scale=patch, norm=False)

        self.positions = positions
        self.pos = modules.Sinusoid(
            scale=pos_scale, global_scale=global_scale)
        self.readout = modules.Readout(d_model=dim)

        self.dec_layers = dec_layers
        self.layers = nn.ModuleList([
            modules.TransformerLayer(
                d_feedforward=int(ff_ratio * dim), d_model=dim,
                n_head=dim // head_dim, dropout=dropout, activation=activation)
            for _ in range(layers)])

    def forward(
        self, x: Float[Tensor, "n d a e r c"]
    ) -> Float[Tensor, "n s c"] | list[Float[Tensor, "n s c"]]:
        """Apply radar transformer.

        Args:
            x: input batch, with batch-doppler-azimuth-elevation-range-iq axis
                order.

        Returns:
            Encoding output.
        """
        embedded = self.patch(x)

        if self.positions == "nd":
            embedded = self.pos(embedded)
        flat = rearrange(embedded, "n d r a e c -> n (d r a e) c")
        if self.positions == "flat":
            flat = self.pos(flat)

        x = self.readout(flat)

        if self.dec_layers is None:
            for layer in self.layers:
                x = layer(x)
            return x
        else:
            out = []
            for i, layer in enumerate(self.layers):
                x = layer(x)
                if i in self.dec_layers:
                    out.append(x)
            return out


class TransformerDecoder(L.LightningModule):
    """Radar transformer tensor decoder.

    Args:
        key: target key, e.g. `bev`, `depth`.
        layers: number of decoder layers.
        dim: hidden dimension; should be the same as the encoder.
        ff_ratio: expansion ratio for feedforward blocks.
        head_dim: number of feature dimensions per head.
        dropout: dropout during training.
        activation: activation function to use.
        shape: output shape; should be a 2 element list or tuple.
        pos_scale: position embedding scale (i.e. the spatial range that this
            axis corresponds to). If `None`, only the global scale is used.
        global_scale: scalar constant to multiply scale by for convenience of
            representation; yields a net scale of `scale * global_scale`.
        patch: patch size to use for unpatching. Must evenly divide `shape`.
        out_dim: output channels; if `=0`, the dimension is omitted entirely,
            i.e. `(h, w)` instead of `(h, w, c)`.
        positions: type of positional embedding. `flat`: flattened positional
            embeddings, similar to the original ViT; `nd`: n-dimensional
            embeddings, splitting the input features into `d` equal chunks
            encoding each axis separately.
        mode: how to obtain the query vector. Can be `last` (use the last
            token, nominally a output token) or `pool` (average pooling).
    """

    def __init__(
        self, key: str, layers: int = 3, dim: int = 768,
        ff_ratio: float = 4.0, head_dim: int = 64, dropout: float = 0.1,
        activation: str = 'GELU', shape: Sequence[int] = (1024, 256),
        pos_scale: Optional[Sequence[float]] = None, global_scale: float = 1.0,
        patch: Sequence[int] = (16, 16), out_dim: int = 0,
        positions: Literal["flat", "nd"] = "flat",
        mode: Literal["last", "pool"] = "last"
    ) -> None:
        super().__init__()

        self.key = key
        self.out_dim = out_dim
        self.mode = mode

        self.layers = nn.ModuleList([
            modules.TransformerDecoder(
                d_feedforward=int(ff_ratio * dim), d_model=dim,
                n_head=dim // head_dim, dropout=dropout, activation=activation)
            for _ in range(layers)])

        query_shape = [s // p for s, p in zip(shape, patch)]
        if positions == "flat":
            query_shape = [int(np.prod(query_shape))]
        self.query = modules.BasisChange(
            shape=query_shape, scale=pos_scale, global_scale=global_scale)

        self.unpatch = modules.Unpatch(
            output_size=(*shape, max(1, self.out_dim)),
            features=dim, size=patch)

    def forward(
        self, encoded: Float[Tensor, "n s c"]
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
        if self.mode == "last":
            x = encoded[:, -1, :]
        else:
            x = torch.mean(encoded, dim=1)

        x = self.query(x)
        enc = encoded[:, :-1, :]
        for layer in self.layers:
            x = layer(x, enc)

        out = self.unpatch(x)
        if self.out_dim == 0:
            out = out[..., 0]

        return {self.key: out}


class VectorDecoder(L.LightningModule):
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
        self, encoded: Float[Tensor, "n s c"]
            | Sequence[Float[Tensor, "?n ?*s ?c"]]
    ) -> dict[str, Float[Tensor, "n f"]]:
        """Apply decoder.

        Args:
            encoded: encoded values. Only the last token (nominally the readout
                token) is used; if a list is passed, the last token of the last
                tensor is used.

        Returns:
            A tensor with the specified number of features.
        """
        if not isinstance(encoded, Tensor):
            encoded = encoded[0]

        if self.channels_first:
            n, c, *s = encoded.shape
            x = encoded.reshape(n, c, -1).permute(0, 2, 1)
        else:
            n, *s, c = encoded.shape
            x = encoded.reshape(n, -1, c)

        if self.strategy == "last":
            x = x[:, -1]
        elif self.strategy == "maxpool":
            x = torch.max(x, dim=1).values
        elif self.strategy == "avgpool":
            x = torch.mean(x, dim=1)
        else:
            raise ValueError(f"Invalid strategy: {self.strategy}")

        return {self.key: self.mlp(x)}
