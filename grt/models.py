"""GRT building blocks.

??? tip "Use `TransformerEncoder` as the GRT encoder"

    The GRT encoder is just a vanilla transformer encoder! If you're using
    hydra, use the following:
    ```yaml
    encoder:
      _target_: torch.nn.TransformerEncoder
      encoder_layer:
        _target_: torch.nn.TransformerEncoderLayer
        d_model: 512
        nhead: 8
        dim_feedforward: 2048
        dropout: 0.1
        activation: gelu
        layer_norm_eps: 1e-5
        batch_first: true
        norm_first: true
        bias: true
      num_layers: 4
      enable_nested_tensor: false
    ```
    To match the settings used by GRT, after selecting the appropriate
    `d_model` and `num_layers`:

    - Set `dim_feedforward` to `4.0 * d_model`
    - Set `n_head` to `d_model // 64`
"""

from collections.abc import Sequence
from typing import Literal

import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor, nn

from nrdk import modules
from nrdk.roverd import SpectrumData


class SpectrumTokenizer(nn.Module):
    """GRT 4D Radar Spectrum tokenizer.

    Two types of positional embeddings are supported:

    - `nd` (recommended): n-dimensional embeddings, splitting the input
        features into `d` equal chunks encoding each axis separately.
    - `flat`: flattened positional embeddings, similar to the original ViT.

    !!! info

        We use a relative coordinate system for positional embeddings instead
        of absolute position indices, where each axis is scaled to `[-1, 1]` by
        default and scaled by `pos_scale` and `global_scale` factors; see
        [`modules.Sinusoid`][nrdk.] for details.

    !!! warning

        Any axes specified in `squeeze` must have a patch size which is equal
        to the input size along that axis.

    Args:
        d_model: model feature dimension.
        patch: input (doppler, azimuth, elevation, range) patch size.
        squeeze: eliminate these axes by moving them to the channel axis prior
            to patching; specified by index.
        n_channels: number of input channels; see [`xwr.nn`][xwr.nn].
        pos_scale: per-axis position embedding scale.
        global_scale: global positional embedding scale.
        positions: type of positional embedding.
    """

    def __init__(
        self, d_model: int = 768, patch: Sequence[int] = (1, 2, 2, 8, 4),
        squeeze: Sequence[int] = [], n_channels: int = 2,
        pos_scale: Sequence[float] | None = None, global_scale: float = 16.0,
        positions: Literal["flat", "nd"] = "nd",
    ) -> None:
        super().__init__()

        if len(patch) != 5:
            raise ValueError(
                f"Invalid patch size: {patch}; expected 5 dims "
                f"(time, doppler, elevation, azimuth, range)")

        if len(squeeze) > 0:
            self.squeeze = modules.Squeeze(dim=squeeze, size=patch)
            n_channels = n_channels * self.squeeze.n_channels
            patch = [p for i, p in enumerate(patch) if i not in squeeze]
        else:
            self.squeeze = None

        self.patch = modules.PatchMerge(
            d_in=n_channels, d_out=d_model, scale=patch, norm=False)

        self.positions = positions
        self.pos = modules.Sinusoid(
            scale=pos_scale, global_scale=global_scale)
        self.readout = modules.Readout(d_model=d_model)

    def forward(
        self, spectrum: SpectrumData
    ) -> Float[Tensor, "n s c"]:
        """Apply radar transformer.

        Args:
            spectrum: input batch spectrum data.

        Returns:
            Tokenized output.
        """
        x = spectrum.spectrum

        if self.squeeze is not None:
            x = self.squeeze(x)

        embedded = self.patch(x)

        if self.positions == "nd":
            embedded = self.pos(embedded)
        flat = embedded.reshape(embedded.shape[0], -1, embedded.shape[-1])
        if self.positions == "flat":
            flat = self.pos(flat)

        return self.readout(flat)


class TransformerDecoder(nn.Module):
    """GRT tensor decoder.

    Args:
        decoder_layer: transformer decoder layer object to use.
        num_layers: number of decoder layers.
        d_model: hidden dimension; should be the same as the encoder.
        shape: output shape; should be a 2 element list or tuple.
        pos_scale: position embedding scale (i.e. the spatial range that this
            axis corresponds to). If `None`, only the global scale is used.
        global_scale: scalar constant to multiply scale by for convenience of
            representation; yields a net scale of `scale * global_scale`.
        patch: patch size to use for unpatching. Must evenly divide `shape`.
        out_dim: output channels; if `=0`, the dimension is omitted entirely,
            i.e. `(h, w)` instead of `(h, w, c)`.
    """

    def __init__(
        self, decoder_layer: nn.TransformerDecoderLayer, d_model: int = 512,
        num_layers: int = 4, shape: Sequence[int] = (1024, 256),
        pos_scale: Sequence[float] | None = None, global_scale: float = 1.0,
        patch: Sequence[int] = (16, 16), out_dim: int = 0,
    ) -> None:
        super().__init__()

        self.out_dim = out_dim

        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers)

        query_shape = [s // p for s, p in zip(shape, patch)]
        self.query = modules.BasisChange(
            shape=query_shape, scale=pos_scale, global_scale=global_scale)

        self.unpatch = modules.Unpatch(
            output_size=(*shape, max(1, self.out_dim)),
            features=d_model, size=patch)

    def forward(
        self, encoded: Float[Tensor, "n s c"]
    ) -> Float[Tensor, "n h w ..."]:
        """Apply decoder.

        Args:
            encoded: list of encoded values. Each tensor should be the same
                size, and use batch-spatial-channel order. The last spatial
                element of each tensor should correspond to a readout token.

        Returns:
            n-dimensional output; only a single key (e.g. the specified `key`)
                is decoded.
        """
        x = self.query(encoded[:, -1, :])
        enc = encoded[:, :-1, :]
        out = self.unpatch(self.decoder(x, enc))

        if self.out_dim == 0:
            out = out[..., 0]
        return out


def _get_activation_fn(activation: str) -> nn.Module:
    """Workaround for naming inconsistency in pytorch.

    Pytorch has an annoying inconsistency with activation names, and manually
    fetches from `nn.functional` in
    `TransformerEncoderLayer/TransformerDecoderLayer`. To match that behavior,
    we manually reproduce it here.
    """
    if activation == "relu":
        return nn.ReLU()
    elif activation == "gelu":
        return nn.GELU()

    raise RuntimeError(f"activation should be relu/gelu, not {activation}")


class VectorDecoder(nn.Module):
    """Generic MLP-based vector decoder without spatial dimensions.

    Always uses the first encoder output tensor, and supports the following
    reduction strategies for that tensor:

    - `last`: take the last spatial feature (nominally a readout token).
    - `max`, `avg`: max or average pooling over all spatial dimensions.

    Args:
        layers: MLP architecture.
        dropout: dropout ratio during training.
        activation: activation function; specify as a name (i.e. corresponding
            to a class in `torch.nn`).
        dim: input features.
        shape: output shape.
        channels_first: whether input channels are in channels-spatial (`NCHW`)
            order instead of spatial-channels order (`NHWC`).
    """

    def __init__(
        self, layers: list[int] = [512, 512], dropout: float = 0.1,
        activation: str = 'GELU', dim: int = 768, shape: Sequence[int] = [4],
        strategy: Literal["last", "maxpool", "avgpool"] = "last",
        channels_first: bool = False
    ) -> None:
        super().__init__()

        self.strategy = strategy
        self.channels_first = channels_first
        self.shape = shape

        _layers = []
        for d1, d2 in zip(([dim] + layers)[:-1], layers):
            _layers += [
                nn.Linear(d1, d2, bias=True),
                _get_activation_fn(activation),
                nn.Dropout(dropout)]
        _layers.append(nn.Linear(([dim] + layers)[-1], np.prod(shape).item()))
        self.mlp = nn.Sequential(*_layers)

    def forward(
        self, encoded: Float[Tensor, "n s c"]
            | Sequence[Float[Tensor, "?n ?*s ?c"]]
    ) -> Float[Tensor, "n f"]:
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

        return self.mlp(x).reshape(x.shape[0], *self.shape)
