"""Nonstandard 2D Convolutional modules."""

from typing import Literal

import torch
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
        padding_mode: padding mode for the depthwise convolution; see
            [`torch.nn.Conv2d`][torch.nn.Conv2d] for details.
        layer_scale_init_value: initial value for layer scaling; if <= 0,
            layer scaling is disabled.
    """

    def __init__(
        self, channels: int, expansion_ratio: int | float = 4.0,
        padding_mode: Literal[
            'zeros', 'reflect', 'replicate', 'circular'] = "zeros",
        layer_scale_init_value: float = 1e-6
    ) -> None:
        super().__init__()

        d = int(channels * expansion_ratio)

        self.dw = nn.Conv2d(
            channels, channels, kernel_size=7, padding=3, groups=channels,
            padding_mode=padding_mode)
        self.norm = nn.LayerNorm(channels)
        self.act = nn.GELU()
        self.pw1 = nn.Conv2d(channels, d, kernel_size=1)
        self.pw2 = nn.Conv2d(d, channels, kernel_size=1)
        self.gamma = nn.Parameter(
            layer_scale_init_value * torch.ones((channels)), requires_grad=True
        ) if layer_scale_init_value > 0 else None

    def forward(self, x: Float[Tensor, "b c h w"]) -> Float[Tensor, "b c h w"]:
        """Forward pass through the ConvNext layer."""
        x1 = self.dw(x)
        x1 = self.norm(x1.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()
        x1 = self.pw2(self.act(self.pw1(x1)))
        if self.gamma is not None:
            x1 = self.gamma[None, :, None, None] * x1
        return x + x1
