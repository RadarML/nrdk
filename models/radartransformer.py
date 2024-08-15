"""First prototype of the "Radar Transformer"."""

import torch
from torch import Tensor, nn
from einops import rearrange

from beartype.typing import Union, cast, Optional
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


class RadarTransformer(nn.Module):
    """U-shape Radar Doppler Transformer.

    Only a selected subset of skip connections are formed; the decoder layers
    are specified by `dec_layers` as follows:

    - Each entry in the list indicates the index of the encoder layer that the
      decoder layer with this index connects to.
    - Indexing starts from 0 at the output of the patch projection (e.g. the
      output of the first encoder layer is index 1)

    Args:
        enc_layers: number of encoder layers.
        dec_layers: decoder layers with skip connection indices.
        dim: hidden dimension.
        ff_ratio: expansion ratio for feedforward blocks.
        heads: number of heads for multiheaded attention.
        dropout: dropout ratio during training.
        activation: activation function; specify as a name (i.e. corresponding
            to a class in `torch.nn`).
        output_activation: activation function to apply to the output; specify
            as a name. If `None`, no activation is applied.
        out_shape: output spatial dimensions.
        patch_size: input range-doppler patch size.
        unpatch_size: output patch size.
    """

    def __init__(
        self, enc_layers: int = 5, dec_layers: list[int] = [5, 3, 1],
        dim: int = 768, ff_ratio: float = 4.0, heads: int = 12,
        dropout: float = 0.1, activation: str = 'GELU',
        output_activation: Optional[str] = None,
        out_shape: Shape2 = (1024, 256),
        patch_size: Shape2 = (16, 16), unpatch_size: Shape2 = (16, 16)
    ) -> None:
        super().__init__()

        patch_size = assert_shape2(patch_size)
        unpatch_size = assert_shape2(unpatch_size)
        out_shape = assert_shape2(out_shape)
        self.dec_layers = dec_layers

        self.patch = modules.Patch4D(channels=2, features=dim, size=patch_size)
        self.pos = modules.Sinusoid()

        self.readout = nn.Parameter(data=torch.normal(0, 0.02, (dim,)))

        self.encode = nn.ModuleList([
            modules.TransformerLayer(
                d_feedforward=int(ff_ratio * dim), d_model=dim, n_head=heads,
                dropout=dropout, activation=activation)
            for _ in range(enc_layers)])
        self.decode = nn.ModuleList([
            modules.TransformerDecoder(
                d_feedforward=int(ff_ratio * dim), d_model=dim, n_head=heads,
                dropout=dropout, activation=activation)
            for _ in dec_layers])

        self.query = BasisChange(shape=(
            out_shape[0] // unpatch_size[0], out_shape[1] // unpatch_size[1]))

        self.unpatch = modules.Unpatch2D(
            output_size=(out_shape[0], out_shape[1], 1),
            features=dim, size=unpatch_size)

        if output_activation is not None:
            self.activation = getattr(nn, output_activation)()
        else:
            self.activation = None

    def forward(
        self, x: Float[Tensor, "n d a e r c"]
    ) -> Float[Tensor, "n a r"]:
        """Apply radar transformer.

        Args:
            x: input batch, with batch-doppler-azimuth-elevation-range-iq axis
                order.

        Returns:
            2-dimensional output, nominally in batch-azimuth-range order
            (though could also be a batch-azimuth-elevation representation).
        """
        patch = self.patch(rearrange(x, "n d a e r c -> n c d r a e"))
        embedded = self.pos(patch)

        x0 = rearrange(embedded, "n d r a e c -> n (d r a e) c")
        readout = torch.tile(self.readout[None, None, :], (x0.shape[0], 1, 1))
        # The output type of `rearrange` isn't inferred correctly.
        x0 = torch.concatenate([x0, readout], axis=1)  # type: ignore

        encoded = [x0]
        for layer in self.encode:
            encoded.append(layer(encoded[-1]))

        out = self.query(encoded[-1][:, -1, :])
        for i, layer in zip(self.dec_layers, self.decode):
            out = layer(out, encoded[i])

        unpatch = self.unpatch(out)[:, 0]
        return unpatch
