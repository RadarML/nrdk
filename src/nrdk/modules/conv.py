"""Nonstandard 2D Convolutional modules."""

from jaxtyping import Float
from torch import Tensor, nn


class ConvNextLayer(nn.Module):
    """Residual convolutional layer following the ConvNext architecture.

    See ["A ConvNet for the 2020s"](https://arxiv.org/abs/2201.03545); each
    residual block consists of:

    1. A 7x7 depthwise convolution,
    2. Layer norm,
    3. A 1x1 "inverted bottleneck" with an expansion ratio of 4,
    4. GELU activation,
    5. And finally a 1x1 pointwise convolution.

    Args:
        channels: number of input/output channels in the model.
        expansion_ratio: expansion ratio for the inverted bottleneck.
    """

    def __init__(
        self, channels: int, expansion_ratio: int | float = 4.0
    ) -> None:
        super().__init__()

        d = int(channels * expansion_ratio)

        self.dw = nn.Conv2d(
            channels, channels, kernel_size=7, padding=3, groups=channels)
        self.norm = nn.LayerNorm(channels)
        self.act = nn.GELU()
        self.pw1 = nn.Conv2d(channels, d, kernel_size=1)
        self.pw2 = nn.Conv2d(d, channels, kernel_size=1)

    def forward(self, x: Float[Tensor, "b c h w"]) -> Float[Tensor, "b c h w"]:
        """Forward pass through the ConvNext layer."""
        x1 = self.dw(x)
        x1 = self.norm(x1.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x1 = self.pw2(self.act(self.pw1(x1)))
        return x + x1
