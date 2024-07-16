"""Model performance metrics."""

import numpy as np
import torch
from torch import Tensor
from torch.nn.functional import max_pool2d, binary_cross_entropy

from beartype.typing import cast, Union
from jaxtyping import Bool, Float


#: Optionally reduced training metric
MetricValue = Union[Float[Tensor, ""], Float[Tensor, "batch"]]


class LPDepth:
    """Generic lp depth loss, with missing value masking.

    Args:
        ord: loss order; only l1 (`1`) and l2 (`2`) are supported.
    """

    def __init__(self, ord: int = 1) -> None:
        self.ord = ord

    def __call__(
        self, y_hat: Float[Tensor, "batch azimuth range"],
        y_true: Float[Tensor, "batch azimuth range"], reduce: bool = True
    ) -> MetricValue:

        if self.ord == 1:
            diff = torch.abs(y_hat - y_true)
        elif self.ord == 2:
            diff = y_hat * y_hat - y_true * y_true
        else:
            raise NotImplementedError("Only L1 and L2 losses are implemented.")

        mask = (y_true != 0)
        res = torch.sum(diff * mask, dim=(1, 2)) / torch.sum(mask, dim=(1, 2))
        if reduce:
            res = torch.mean(res)        
        return res


class CombinedDiceBCE:
    """Weighted combination of Dice and BCE loss.

    Supports equal-area "range weighting," where the relative weight of bins
    is adjusted on the range axis so that each bin is weighted according to
    the area which it represents (i.e. multiplying the weight by its range).

    Args:
        bce_weight: BCE loss weight; Dice loss is weighted `1 - bce_weight`.
        range_weighted: perform range equal-area weighting if `True`.
    """

    def __init__(
        self, bce_weight: float = 0.9, range_weighted: bool = False
    ) -> None:
        self.bce_weight = bce_weight
        self.range_weighted = range_weighted

    @staticmethod
    def _dice(
        y_hat: Float[Tensor, "batch azimuth range"],
        y_true: Float[Tensor, "batch azimuth range"], weight
    ) -> Float[Tensor, "batch"]:
        denominator = (
            torch.sum(y_hat * y_hat * weight, dim=(1, 2))
            + torch.sum(y_true * weight, dim=(1, 2)))
        numerator = 2 * torch.sum(y_hat * y_true * weight, dim=(1, 2))
        return 1.0 - numerator / denominator

    def __call__(
        self, y_hat: Float[Tensor, "batch azimuth range"],
        y_true: Float[Tensor, "batch azimuth range"], reduce: bool = True
    ) -> MetricValue:
        """Get Dice + BCE weighted loss."""

        if self.range_weighted:
            bins = torch.arange(y_true.shape[2], device=y_hat.device)
            weight = ((bins + 1) / y_true.shape[2])[None, None, :]
        else:
            # Mypy doesn't seem to recognize the type overloading here.
            weight = 1.0  # type: ignore

        dice = self._dice(y_hat, y_true, weight)
        bce = torch.mean(
            weight * binary_cross_entropy(y_hat, y_true, reduction='none'),
            dim=(1, 2))

        loss = bce * self.bce_weight + dice * (1 - self.bce_weight)
        if reduce:
            loss = torch.mean(loss)
            
        return loss


class Chamfer:
    """L2 Chamfer distance for a polar range-azimuth grid, in range bins.

    Supported modes:

    - `chamfer` (default): chamfer distance (mean)
    - `hausdorff`: hausdorff distance (max)
    - `modhausdorff`: modified hausdorff distance (median)

    Args:
        mode: specified modes.
        on_empty: value to use if one of the provided maps is completely empty.
    """

    def __init__(self, on_empty: float = 64.0, mode: str = "chamfer") -> None:
        self.mode = mode
        self.on_empty = on_empty

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
        self, y_hat: Bool[Tensor, "b w h"], y_true: Bool[Tensor, "b w h"],
        reduce: bool = True
    ) -> MetricValue:
        """Compute chamfer distance, in range bins."""

        def _forward(x, y):
            pts_x = self.as_points(x)
            pts_y = self.as_points(y)
            dist = self.distance(pts_x, pts_y)

            if dist.shape[0] == 0 or dist.shape[1] == 0:
                return torch.full((), self.on_empty)

            d1 = torch.min(dist, dim=0).values
            d2 = torch.min(dist, dim=1).values

            if self.mode == "hausdorff":
                m1 = torch.max(d1, dim=1).values
                m2 = torch.max(d2, dim=1).values
                return torch.mean(torch.maximum(m1, m2))
            elif self.mode == "modhausdorff":
                m1 = torch.mean(torch.median(d1))
                m2 = torch.mean(torch.median(d2))
                return (m1 + m2) / 2
            else:
                return (torch.mean(d1) + torch.mean(d2)) / 2

        _iter = (_forward(xs, ys) for xs, ys in zip(y_hat, y_true))

        if reduce:
            return cast(Float[Tensor, ""], sum(_iter)) / y_hat.shape[0]
        else:
            return torch.Tensor(list(_iter))


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
        self, y_hat: Bool[Tensor, "b w h"], y_true: Bool[Tensor, "b w h"],
        reduce: bool = True
    ) -> Float[Tensor, ""]:
        """Compute grid chamfer distance, in grid units."""
        assert reduce
        d1 = torch.mean(self.as_distance(y_true) * y_hat)
        d2 = torch.mean(self.as_distance(y_hat) * y_true)
        return (d1 + d2) / 2
