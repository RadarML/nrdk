"""RadarHD U-net model."""

import torch
from torch import Tensor

from jaxtyping import Float, Complex

from torch import nn
from deepradar import modules


def _unetblock(
    d_in: int = 64, d_out: int = 64, d_hidden: int = 64
) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(d_in, d_hidden, kernel_size=3, padding='same'),
        nn.BatchNorm2d(d_hidden),
        nn.ReLU(inplace=True),
        nn.Conv2d(d_hidden, d_out, kernel_size=3, padding='same'),
        nn.BatchNorm2d(d_out),
        nn.ReLU(inplace=True))


def _down(d_in: int = 64, d_out: int = 64) -> nn.Module:
    return nn.Sequential(nn.MaxPool2d(2), _unetblock(d_in, d_out, d_out))


class UNetUp(nn.Module):
    """Single upsample U-net stage."""

    def __init__(self, d_in: int = 64, d_out: int = 64) -> None:
        super().__init__()

        self.up = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = _unetblock(d_in, d_out, d_in // 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = nn.functional.pad(x1, [
            diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class AzimuthUp(nn.Module):
    """Upsample azimuth axis."""

    def __init__(self, d_in: int = 64, d_out: int = 64) -> None:
        super().__init__()
        self.up = nn.Upsample(
            scale_factor=(2, 1), mode='bilinear', align_corners=True)
        self.conv = _unetblock(d_in, d_out, d_in)

    def forward(self, x1):
        x1 = self.up(x1)
        return self.conv(x1)


class RadarUNet(nn.Module):
    """Generic U-net for range-azimuth radar [N1]_.

    Args:
        d_input: Number of input features (i.e. slow time / doppler length).
        doppler: Whether to run FFT on the doppler axis, i.e. use explicit
            doppler information.
    """

    def __init__(self, d_input: int = 64, doppler: bool = False) -> None:
        super().__init__()

        axes = (0, 1, 2) if doppler else (1, 2)
        self.fft = modules.FFTLinear(pad=56, axes=axes)

        self.inc = _unetblock(d_input, 64)
        self.down1 = _down(64, 128)
        self.down2 = _down(128, 256)
        self.down3 = _down(256, 512)
        self.down4 = _down(512, 512)
        self.up1 = UNetUp(1024, 256)
        self.up2 = UNetUp(512, 128)
        self.up3 = UNetUp(256, 64)
        self.up4 = UNetUp(128, 64)
        self.up5 = AzimuthUp(64, 64)
        self.up6 = AzimuthUp(64, 64)
        self.up7 = AzimuthUp(64, 64)
        self.up8 = AzimuthUp(64, 64)

        self.out = nn.Conv2d(64, 1, kernel_size=1)
        self.logits = nn.Sigmoid()

    def forward(
        self, x: Complex[Tensor, "n d 4 2 r"]
    ) -> Float[Tensor, "n 1024 256"]:

        x = torch.sqrt(torch.abs(self.fft(x))) / 1e3

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.up5(x)
        x = self.up6(x)
        x = self.up7(x)
        x = self.up8(x)

        return self.logits(self.out(x)[:, 0])
