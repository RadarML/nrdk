"""Axial Transformer."""

from beartype.typing import Literal, Optional, Sequence
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn

from deepradar import modules


class AxialEncoder(nn.Module):
    """Radar Axial Transformer.

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
        heads: number of heads for multiheaded attention.
        dropout: dropout ratio during training.
        activation: activation function; specify as a name (i.e. corresponding
            to a class in `torch.nn`).
        patch: input (doppler, azimuth, elevation, range) patch size.
        positions: type of positional embedding. `flat`: flattened positional
            embeddings, similar to the original ViT; `nd`: n-dimensional
            embeddings, splitting the input features into `d` equal chunks
            encoding each axis separately.
    """

    def __init__(
        self, layers: int = 5, dec_layers: Optional[Sequence[int]] = None,
        dim: int = 768, ff_ratio: float = 4.0, heads: int = 12,
        dropout: float = 0.1, activation: str = 'GELU',
        patch: Sequence[int] = (16, 1, 1, 16),
        size: Sequence[int] = (64, 8, 2, 256),
        windows: Sequence[Sequence[int]] = [[8, 8, 1, 1], [1, 1, 2, 16]],
        positions: Literal["flat", "nd"] = "flat"
    ) -> None:
        super().__init__()

        if len(patch) != 4:
            raise ValueError("Must specify a 4D patch size.")

        self.patch = modules.PatchMerge(
            d_in=2, d_out=dim, scale=patch, norm=False)

        self.positions = positions
        self.pos = modules.Sinusoid()

        self.dec_layers = dec_layers
        self.layers = nn.ModuleList([
            modules.AxialTransformerLayer(
                d_feedforward=int(ff_ratio * dim), d_model=dim, n_head=heads,
                dropout=dropout, activation=activation, windows=windows,
                size=[s // p for s, p in zip(size, patch)],
                mode="compose")
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
        x = self.pos(self.patch(x))
        if self.dec_layers is None:
            for layer in self.layers:
                x = layer(x)
            return rearrange(x, "n d a e r c -> n (d a e r) c")
        else:
            out = []
            for i, layer in enumerate(self.layers):
                x = layer(x)
                if i in self.dec_layers:
                    out.append(rearrange(x, "n d a e r c -> n (d a e r) c"))
            return out
