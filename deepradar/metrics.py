"""Model performance metrics."""

import numpy as np
import torch
from torch import Tensor
from torch.nn.functional import max_pool2d, binary_cross_entropy

from beartype.typing import cast
from jaxtyping import Bool, Float


class CombinedDiceBCE:
    """Weighted combination of Dice and BCE loss.
    
    Args:
        bce_weight: BCE loss weight; Dice loss is weighted `1 - bce_weight`.
    """

    def __init__(self, bce_weight: float = 0.9) -> None:
        self.bce_weight = bce_weight
    
    def __call__(
        self, y_hat: Float[Tensor, "b w h"], y_true: Float[Tensor, "b w h"]
    ) -> Float[Tensor, ""]:
        """Get Dice + BCE weighted loss."""
        denominator = (
            torch.sum(y_hat * y_hat, dim=(1, 2))
            + torch.sum(y_true, dim=(1, 2)))
        numerator = 2 * torch.sum(y_hat * y_true, dim=(1, 2))
        dice = 1.0 - torch.mean(numerator / denominator)

        bce = binary_cross_entropy(y_hat, y_true, reduction='mean')
        return bce * self.bce_weight + dice * (1 - self.bce_weight)


class PolarChamfer:
    """L2 Chamfer distance for a polar range-azimuth grid."""

    @staticmethod
    def as_points(mask: Bool[Tensor, "azimuth range"]) -> Float[Tensor, "n 2"]:
        """Convert (azimuth, range) occupancy grid to points."""
        bin, r = torch.where(mask)
        phi = (bin - mask.shape[0] // 2) / mask.shape[0] * np.pi
        x = torch.cos(phi) * r
        y = torch.sin(phi) * r
        return torch.stack([x, y]).T
    
    @staticmethod
    def distance(
        x: Float[Tensor, "n1 d"], y: Float[Tensor, "n2 d"]
    ) -> Float[Tensor, "n1 n2"]:
        """Compute the pairwise distances between x and y."""
        return torch.sqrt(torch.sum(torch.square(
            x[:, None, :] - y[None, :, :]
        ), dim=-1))

    def __call__(
        self, y_hat: Bool[Tensor, "b w h"], y_true: Bool[Tensor, "b w h"]
    ) -> Float[Tensor, ""]:
        """Compute chamfer distance, in range bins."""

        def _forward(x, y):
            pts_x = self.as_points(x)
            pts_y = self.as_points(y)
            dist = self.distance(pts_x, pts_y)

            if dist.shape[0] == 0 or dist.shape[1] == 0:
                return torch.zeros(())

            d1 = torch.mean(torch.min(dist, dim=0).values)
            d2 = torch.mean(torch.min(dist, dim=1).values)
            return (d1 + d2) / 2

        _iter = (_forward(xs, ys) for xs, ys in zip(y_hat, y_true))
        return cast(Float[Tensor, ""], sum(_iter)) / y_hat.shape[0]


class L1Chamfer:
    """L1 Chamfer distance for a 2D occupancy grid.
    
    WARNING: this is very slow, and probably needs to be completely rethought.
    In particular, max_pool2d is probably not the shortcut it should be in
    pytorch, probably due to the lack of JIT here.
    """

    @staticmethod
    def as_distance(mask: Bool[Tensor, "b w h"]) -> Float[Tensor, "b w h"]:
        """Get grid distance transform.
        
        Each element of the output denotes the L1 distance to the nearest
        occupied point in `mask`, measured in grid cells.
        """
        # Need to do this as a float since `max_pool2d` is only implemented
        # for floats in pytorch
        dmax = max(mask.shape[1:])
        mask = mask.to(torch.float32)
        dist = (1 - mask) * dmax

        for i in range(dmax):
            mask = max_pool2d(mask, kernel_size=3, stride=1, padding=1)
            ring = (dist == dmax) * mask
            dist = dist * (1 - ring) + (i + 1) * ring

        return dist

    def __call__(
        self, y_hat: Bool[Tensor, "b w h"], y_true: Bool[Tensor, "b w h"]
    ) -> Float[Tensor, ""]:
        """Compute grid chamfer distance, in grid units."""
        d1 = torch.mean(self.as_distance(y_true) * y_hat)
        d2 = torch.mean(self.as_distance(y_hat) * y_true)
        return (d1 + d2) / 2
