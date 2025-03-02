"""RADDet Architecture."""

import torch
from einops import rearrange
from jaxtyping import Complex, Float
from torch import Tensor, nn

from deepradar import modules


class RADDetEncoder(nn.Module):
    """RADDet encoder.

    Args:
        dim: number of features (must be equal to the number of doppler bins).
    """

    def __init__(
        self, dim: int = 64
    ) -> None:
        super().__init__()

        block1 = nn.Sequential(*(
            [modules.ConvResidual(dim) for _ in range(2)]
            + [nn.MaxPool2d((2, 2))]))
        block2 = nn.Sequential(*(
            [modules.ConvResidual(dim) for _ in range(4)]
            + [nn.MaxPool2d((2, 2))]))
        block3 = nn.Sequential(*(
            [nn.Conv2d(dim, dim * 2, kernel_size=(3, 3), padding="same")]
            + [modules.ConvResidual(dim * 2) for _ in range(8)]
            + [nn.MaxPool2d((2, 2))]))
        block4 = nn.Sequential(*(
            [nn.Conv2d(dim * 2, dim * 4, kernel_size=(3, 3), padding="same")]
            + [modules.ConvResidual(dim * 4) for _ in range(16)]
            + [nn.MaxPool2d((2, 2))]))

        self.network = nn.Sequential(block1, block2, block3, block4)

    def forward(
        self, x: Complex[Tensor, "n d tx rx r"]
    ) -> list[Float[Tensor, "n ..."]]:
        """Apply Network.

        Args:
            x: input batch, with batch-doppler-rx-tx-range axis order.

        Returns:
            Encoded output; note that tensors have different sizes since they
            correspond to different skip connections.
        """
        x = rearrange(x, "n d tx rx r -> n d (tx rx) r")
        zeros = torch.zeros(
            (x.shape[0], x.shape[1], 256 - x.shape[2], x.shape[3]),
            device=x.device)
        padded = torch.concatenate([x, zeros], dim=2)
        cube_dar: Float[Tensor, "n d 256 r"] = torch.abs(
            torch.fft.fftn(padded, dim=(1, 2, 3)))

        output = self.network(cube_dar)
        return output


class RADDetDecoder(nn.Module):
    """RADDet coordinate transform decoder.

    Args:
        key: output key; only supports `bev`.
        dim: model width (i.e. number of features).
    """

    def __init__(self, key: str = "bev", dim: int = 64) -> None:
        super().__init__()

        self.key = key


    def forward(
        self, encoded: list[Float[Tensor, "n ..."]]
    ) -> dict[str, Float[Tensor, "n 1024 256"]]:
        """Apply decoder.

        Args:
            encoded: backbone output.

        Returns:
            Range-azimuth output.
        """
        return {self.key: self.out(x)}
