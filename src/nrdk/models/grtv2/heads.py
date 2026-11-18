"""GRTv2 task heads."""

from collections.abc import Sequence

from jaxtyping import Float
from torch import Tensor, nn

from nrdk.modules import Unpatch


class LinearHead(nn.Module):
    """Linear projection head for GRTv2 decoder output.

    Applies a linear projection and reconstructs the full spatial output grid.
    Each decoder query token predicts one spatial patch of values; the head
    reassembles patches into the original image shape.

    Args:
        d_model: input feature dimension from the decoder.
        out_channels: number of predicted output channels per voxel.
        output_size: full output spatial shape (e.g. [el, az, rng]), without
            the channel axis.
        patch_size: spatial patch size per axis. If an int, applied uniformly
            across all spatial axes.
        squeeze: if `True`, drop the trailing channel axis from the output;
            only valid when `out_channels=1`.
    """

    def __init__(
        self, d_model: int, out_channels: int,
        output_size: Sequence[int],
        patch_size: int | Sequence[int] = 1,
        squeeze: bool = False,
    ) -> None:
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = [patch_size] * len(output_size)
        self.squeeze = squeeze
        self.unpatch = Unpatch(
            output_size=[*output_size, out_channels],
            features=d_model,
            size=list(patch_size),
        )

    def forward(
        self, x: Float[Tensor, "batch n d_model"]
    ) -> Float[Tensor, "batch ..."]:
        """Project decoder outputs and reassemble the spatial output grid."""
        out = self.unpatch(x)
        if self.squeeze:
            out = out.squeeze(-1)
        return out
