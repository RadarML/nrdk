"""RadarHD U-net model."""

import torch
from jaxtyping import Float
from torch import Tensor, nn
from einops import rearrange


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


def _down(
    d_in: int = 64, d_out: int = 64, size: int | tuple[int, int] = 2
) -> nn.Module:
    return nn.Sequential(nn.MaxPool2d(size), _unetblock(d_in, d_out, d_out))


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

        # 64C x 64D x 256R
        self.inc = _unetblock(n_doppler, dim * 1)  # -> 64C x 64A x 256R
        self.down1 = _down(dim * 1, dim * 2)  # -> 128C x 32A x 128R
        self.down2 = _down(dim * 2, dim * 4)  # -> 256C x 16A x 64R
        self.down3 = _down(dim * 4, dim * 8)  # -> 512C x 8A x 32R
        self.down4 = _down(dim * 8, dim * 8)  # -> 512C x 4A x 16R

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

        # (512 + 512)C x 4D x 16R
        self.up1 = UNetUp(dim * 16, dim * 4)  # -> 256C x 8A x 32R
        self.up2 = UNetUp(dim * 8, dim * 2)  # -> 128C x 16A x 64R
        self.up3 = UNetUp(dim * 4, dim)  # -> 64C x 32A x 128R
        self.up4 = UNetUp(dim * 2, dim)  # -> 64C x 64A x 256R
        self.up5 = AzimuthUp(dim, dim)  # -> 64C x 128A x 256R
        self.up6 = AzimuthUp(dim, dim)  # -> 64C x 256A x 256R
        self.up7 = AzimuthUp(dim, dim)  # -> 64C x 512A x 256R
        self.up8 = AzimuthUp(dim, dim)  # -> 64C x 1024A x 256R

        self.out = nn.Conv2d(64, 1, kernel_size=1)  # -> 1C x 1024A x 256R


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


class UNet3DEncoder(nn.Module):
    """Generic U-net encoder for range-azimuth-elevation radar [N1]_.

    NOTE: must be paired with a :py:class:`.UNetBEVDecoder`.

    Args:
        dim: model base number of features.
        n_doppler: number of doppler bins.
    """

    def __init__(
        self, dim: int = 64, n_doppler: int = 64, n_elev: int = 2
    ) -> None:
        super().__init__()

        # (64 x 2)C x 32A x 256R
        self.inc = _unetblock(n_doppler * n_elev, dim)  # -> 64C x 32A x 256R
        self.down1 = _down(dim, dim * 2, size=(1, 2))  # -> 128C x 32A x 128R
        self.down2 = _down(dim * 2, dim * 4, size=(1, 2))  # -> 256C x 32A x 64R
        self.down3 = _down(dim * 4, dim * 8)  # -> 512C x 16A x 32R
        self.down4 = _down(dim * 8, dim * 8)  # -> 512C x 8A x 16R

    def forward(
        self, x: Float[Tensor, "n d a e r"]
    ) -> list[Float[Tensor, "n ..."]]:
        """Apply UNet.

        Args:
            x: input batch, with batch-doppler-rx-tx-range axis order.

        Returns:
            Encoded output; note that tensors have different sizes since they
            correspond to different skip connections.
        """
        x = rearrange(x, "n d a e r -> n (d e) a r")

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        return [x5, x4, x3]


class UNet3DDecoder(nn.Module):
    """U-net decoder for range-azimuth-elevation radar [N1]_.

    Args:
        key: output key; only supports `bev`.
        dim: model width (i.e. number of features).
    """

    def __init__(self, key: str = "bev", dim: int = 64) -> None:
        super().__init__()

        self.key = key

        # (512 + 512)C x 8A x 8R
        self.up1 = UNetUp(dim * 16, dim * 4)  # -> 256C x 16A x 32R
        self.up2 = UNetUp(dim * 8, dim * 2)  # -> 128C x 32A x 64R
        self.up3 = AzimuthUp(dim * 2, dim * 2)  # -> 128C x 64A x 64R
        self.up4 = AzimuthUp(dim * 2, dim)  # -> 64C x 128A x 64R
        self.out = nn.Conv2d(64, 64, kernel_size=1)  # -> 64E x 128A x 64R

    def forward(
        self, encoded: list[Float[Tensor, "n ..."]]
    ) -> dict[str, Float[Tensor, "n 128 256"]]:
        """Apply UNet.

        Args:
            encoded: U-net skip connections.

        Returns:
            Range-azimuth output.
        """
        x5, x4, x3 = encoded

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x)
        x = self.up4(x)

        return {self.key: self.out(x)}
