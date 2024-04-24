"""U-net model.

References
----------
[1] RadarHD: High resolution point clouds from mmWave Radar
    https://akarsh-prabhakara.github.io/research/radarhd/
[2] ConvNeXt: A ConvNet for the 2020s
    https://arxiv.org/abs/2201.03545
"""

import torch
from torch import Tensor
from einops import rearrange

from jaxtyping import Float, Complex

from torch import nn
from radar import modules


class UNetDown(nn.Module):
    """Single downsample U-net stage."""

    def __init__(self, width: int = 64, depth: int = 3) -> None:
        super().__init__()

        self.conv = nn.Sequential(*[
            modules.ConvNeXTBlock(
                d_model=width, d_bottleneck=width * 4, kernel_size=7,
                activation="ReLU"
            ) for _ in range(depth)])

        self.down = modules.ConvDownsample(d_in=width, d_out=width * 2)

    def forward(
        self, x: Float[Tensor, "n c h w"]
    ) -> tuple[Float[Tensor, "n c h w"], Float[Tensor, "n c2 h2 h2"]]:
        x = self.conv(x)
        return x, self.down(x)


class UNetUp(nn.Module):
    """Single upsample U-net stage."""

    def __init__(self, width: int = 64, depth: int = 3) -> None:
        super().__init__()

        self.up = nn.PixelShuffle(2)
        self.linear_up = nn.Conv2d(
            width * 2, width * 4, kernel_size=(1, 1), stride=1)

        self.conv = nn.Sequential(*[
            modules.ConvNeXTBlock(
                d_model=width, d_bottleneck=width * 4, kernel_size=7,
                activation="ReLU"
            ) for _ in range(depth)])

    def forward(
        self, x_down: Float[Tensor, "n c2 h w"],
        x_skip: Float[Tensor, "n c h2 w2"]
    ) -> Float[Tensor, "n c h2 w2"]:
        x_upsampled = self.up(self.linear_up(x_down))
        return self.conv(x_upsampled + x_skip)


class AzimuthUp(nn.Module):
    """Upsample azimuth axis."""

    def __init__(self, width: int, depth: int = 3) -> None:
        super().__init__()

        self.conv = nn.Sequential(*[
            modules.ConvNeXTBlock(
                d_model=width, d_bottleneck=width * 4, kernel_size=7,
                activation='ReLU'
            ) for _ in range(depth)])

    def forward(
        self, x: Float[Tensor, "n c h w"]
    ) -> Float[Tensor, "n c2 h2 w"]:
        x_shuffle = rearrange(x, "n (c f) h w -> n c (h f) w", f=2)
        return self.conv(x_shuffle)


class RadarUNet(nn.Module):
    """Radar range-azimuth U-net.

    We adopt some relevant recommendations from [2], including:
    - Using layer norm instead of batch norm.
    - Using the ConvNeXT block (7x7/d + 1x1/4d + 1x1/d) instead of a generic
        3x3/d conv block.

    We also take inspiration from the same concepts and change the following:
    - Instead of concatenating skip connections, we upsample-project to the
        same space, and add (similar to a residual layer).

    Parameters
    ----------
    width: network width multiplier. Each stage has a width of
        `width * 2**stage`, for stage = 0, 1, 2.
    d_input: input dimension.
    depth: number of conv blocks per encode/decode in each stage.
    """

    def __init__(
        self, width: int = 64, d_input: int = 64, depth: int = 3
    ) -> None:
        super().__init__()

        self.embed = nn.Conv2d(d_input, width, kernel_size=(1, 1))
        self.s0_down = UNetDown(width, depth=depth)
        self.s1_down = UNetDown(width * 2, depth=depth)
        self.s2_down = UNetDown(width * 4, depth=depth)

        self.bridge = nn.Sequential(*[
            modules.ConvNeXTBlock(
                d_model=width *  8, d_bottleneck=width * 32, kernel_size=7,
                activation="ReLU"
            ) for _ in range(depth)])

        self.s0_up = UNetUp(width, depth=depth)
        self.s1_up = UNetUp(width * 2, depth=depth)
        self.s2_up = UNetUp(width * 4, depth=depth)

        self.asym_up = nn.Sequential(
            AzimuthUp(width // 2),
            AzimuthUp(width // 4),
            AzimuthUp(width // 8),
            AzimuthUp(width // 16))

        self.out = nn.Conv2d(width // 16, 1, kernel_size=(1, 1))

        self.logits = nn.Sigmoid()

    def forward(
        self, x: Complex[Tensor, "n d 4 2 r"]
    ) -> Float[Tensor, "n 1024 256"]:

        with torch.no_grad():
            x_iq = rearrange(x, "n d tx rx r -> n d (tx rx) r")
            zeros = torch.zeros(
                (x.shape[0], x.shape[1], 56, x.shape[-1]), device=x.device)
            x_pad = torch.concatenate([x_iq, zeros], dim=2)
            x_dar = torch.fft.fftn(x_pad, dim=(2, 3))
            x_shf = torch.fft.fftshift(x_dar, dim=[2])
            x = torch.sqrt(torch.abs(x_shf)) / 1e3

        x = self.embed(x)
        x0, x = self.s0_down(x)
        x1, x = self.s1_down(x)
        x2, x = self.s2_down(x)
        x_bridge = self.bridge(x)
        x = self.s2_up(x_bridge, x2)
        x = self.s1_up(x, x1)
        x = self.s0_up(x, x0)

        return self.logits(self.out(self.asym_up(x))[:, 0])
