"""Generalizable Radar Transformer (Fourier Encoder)."""

import lightning as L
import torch
from beartype.typing import Literal, Optional, Sequence
from jaxtyping import Float
from torch import Tensor, nn

from deepradar import modules


class FourierNetwork(nn.Module):
    """FFT-like network.

    Takes an arbitrary number of spatial axes, along with a batch and trailing
    channels (nominally real/imaginary) axis. For each spatial axis, a separate
    linear transform is applied across that axis and the channels axis.
    """

    def __init__(
        self, shape: Sequence[int] = (64, 12, 256), channels: int = 2
    ) -> None:
        super().__init__()

        self.transforms = nn.ParameterList([
            nn.Parameter(
                torch.nn.init.kaiming_uniform_(
                    torch.empty(channels * x, channels * x, dtype=torch.float))
            ) for x in shape
        ])

    def __call__(
        self, x: Float[Tensor, "batch *spatial channels"]
    ) -> Float[Tensor, "batch *spatial channels"]:
        """Apply FFT-like linear transformations.

        Args:
            x: input, with one leading batch axis, and trailing real/im axis.

        Returns:
            Transformed data.
        """
        _, *spatial, channels = x.shape

        # For each spatial axis:
        for axis, tf in enumerate(self.transforms):
            # Put the spatial axis in the 2nd to last position, then merge
            # the last two axes (spatial * channel)
            x = x.moveaxis(axis + 1, -2)
            _permuted = x.shape
            x = x.reshape(-1, channels * spatial[axis])

            # Apply the linear transform, unshape it, then put it back.
            x = x.matmul(tf).reshape(_permuted)
            x = x.moveaxis(-2, axis + 1)

        return x


class TransformerFourierEncoder(L.LightningModule):
    """Radar 4D Doppler Transformer.

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
        patch: input (doppler, antenna, range) patch size.
        shape: input I/Q data cube shape.
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
        patch: Sequence[int] = (2, 12, 4),
        shape: Sequence[int] = (64, 3, 4, 256),
        pos_scale: Optional[Sequence[float]] = None, global_scale: float = 1.0,
        input_channels: int = 2,
        positions: Literal["flat", "nd"] = "nd",
    ) -> None:
        super().__init__()

        if len(patch) not in {3, 4}:
            raise ValueError(f"Must specify a 3D or 4D patch size ({patch}).")

        D, Tx, Rx, R = shape
        self.fourier = FourierNetwork(
            shape=(D, Tx * Rx, R), channels=input_channels)

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
        self, x: Float[Tensor, "n *t d tx rx r c1"]
    ) -> Float[Tensor, "n s c"] | list[Float[Tensor, "n s c"]]:
        """Apply radar transformer.

        Args:
            x: input batch, with batch-doppler-tx-rx-range-iq (or
                batch-time-... if 5D) axis order.

        Returns:
            Encoding output.
        """
        n, *t, d, tx, rx, r, c = x.shape

        x = x.reshape((-1, d, tx * rx, r, c))
        x = self.fourier(x).reshape((n, *t, d, tx * rx, r, c))

        embedded = self.patch(x)

        if self.positions == "nd":
            embedded = self.pos(embedded)
        flat = embedded.reshape(embedded.shape[0], -1, embedded.shape[-1])
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
