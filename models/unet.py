"""RadarHD U-net model."""

import torch
from jaxtyping import Float
from torch import Tensor, nn


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

        # Handle any odd sizes. If up(x1) and x2 have the exact same size, this
        # step has no effect.
        dy = x2.size()[2] - x1.size()[2]
        dx = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(
            x1, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])

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


class UNetEncoder(nn.Module):
    """Generic U-net encoder for range-azimuth radar [N1]_.

    NOTE: must be paired with a :py:class:`.UNetBEVDecoder`.

    Args:
        dim: model base number of features.
        n_doppler: number of doppler bins.
    """

    def __init__(
        self, dim: int = 64, n_doppler: int = 64
    ) -> None:
        super().__init__()

        self.inc = _unetblock(n_doppler, dim * 1)
        self.down1 = _down(dim * 1, dim * 2)
        self.down2 = _down(dim * 2, dim * 4)
        self.down3 = _down(dim * 4, dim * 8)
        self.down4 = _down(dim * 8, dim * 8)

    def forward(
        self, x: Float[Tensor, "n d a r"]
    ) -> list[Float[Tensor, "n ..."]]:
        """Apply UNet.

        Args:
            x: input batch, with batch-doppler-rx-tx-range axis order.

        Returns:
            Encoded output; note that tensors have different sizes since they
            correspond to different skip connections.
        """
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        return [x5, x4, x3, x2, x1]


class UNetBEVDecoder(nn.Module):
    """Generic U-net decoder for range-azimuth radar [N1]_.

    NOTE: must be paired with a :py:class:`.UNetEncoder`; can only be used for
    range-azmiuth `bev`.

    Args:
        key: output key; only supports `bev`.
        dim: model width (i.e. number of features).
    """

    def __init__(self, key: str = "bev", dim: int = 64) -> None:
        super().__init__()

        self.key = key

        self.up1 = UNetUp(dim * 16, dim * 4)
        self.up2 = UNetUp(dim * 8, dim * 2)
        self.up3 = UNetUp(dim * 4, dim)
        self.up4 = UNetUp(dim * 2, dim)
        self.up5 = AzimuthUp(dim, dim)
        self.up6 = AzimuthUp(dim, dim)
        self.up7 = AzimuthUp(dim, dim)
        self.up8 = AzimuthUp(dim, dim)

        self.out = nn.Conv2d(64, 1, kernel_size=1)


    def forward(
        self, encoded: list[Float[Tensor, "n ..."]]
    ) -> dict[str, Float[Tensor, "n 1024 256"]]:
        """Apply UNet.

        Args:
            encoded: U-net skip connections.

        Returns:
            Range-azimuth output.
        """
        x5, x4, x3, x2, x1 = encoded

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.up5(x)
        x = self.up6(x)
        x = self.up7(x)
        x = self.up8(x)

        return {self.key: self.out(x)}
