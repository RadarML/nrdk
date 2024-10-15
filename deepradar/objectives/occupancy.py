"""Occupancy classification objective."""

import numpy as np
import torch
from beartype.typing import cast
from jaxtyping import Bool, Float, Shaped
from torch import Tensor
from torch.nn.functional import binary_cross_entropy_with_logits, sigmoid

from deepradar.utils import comparison_grid, polar2_to_bev

from .base import Metrics, MetricValue, Objective


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
        y_true: Bool[Tensor, "batch azimuth range"], reduce: bool = True
    ) -> MetricValue:
        """Get Dice + BCE weighted loss.

        Args:
            y_hat: output logits.
            y_true: occupancy grid values.

        Returns:
            Loss value, possibly reduced.
        """
        y_true = y_true.to(y_hat.dtype)

        if self.range_weighted:
            bins = torch.arange(y_true.shape[2], device=y_hat.device)
            weight = ((bins + 1) / y_true.shape[2])[None, None, :]
        else:
            # Mypy doesn't seem to recognize the type overloading here.
            weight = 1.0  # type: ignore

        dice = self._dice(sigmoid(y_hat), y_true, weight)
        bce = torch.mean(
            weight * binary_cross_entropy_with_logits(
                y_hat, y_true, reduction='none'),
            dim=(1, 2))

        loss = bce * self.bce_weight + dice * (1 - self.bce_weight)
        if reduce:
            loss = torch.mean(loss)

        return loss


class Chamfer2D:
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

    def _forward(self, x, y):
        pts_x = self.as_points(x)
        pts_y = self.as_points(y)
        dist = self.distance(pts_x, pts_y)

        if dist.shape[0] == 0 or dist.shape[1] == 0:
            return torch.full((), self.on_empty, device=x.device)

        d1 = torch.min(dist, dim=0).values
        d2 = torch.min(dist, dim=1).values

        if self.mode == "modhausdorff":
            return torch.median(torch.concatenate([d1, d2]))
        elif self.mode == "chamfer":
            return (torch.mean(d1) + torch.mean(d2)) / 2
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    def __call__(
        self, y_hat: Bool[Tensor, "b w h"], y_true: Bool[Tensor, "b w h"],
        reduce: bool = True
    ) -> MetricValue:
        """Compute chamfer distance, in range bins."""
        _iter = (self._forward(xs, ys) for xs, ys in zip(y_hat, y_true))
        if reduce:
            return cast(Float[Tensor, ""], sum(_iter)) / y_hat.shape[0]
        else:
            return torch.stack(list(_iter))


class BEVOccupancy(Objective):
    """Radar -> lidar as bird's eye view (BEV) occupancy.

    Args:
        weight: total objective weight.
        bce_weight: BCE loss weight; Dice loss is weighted `1 - bce_weight`.
        range_weighted: Whether to apply equal-area "range weighting", where
            the relative weight of bins is adjusted on the range axis so that
            each bin is weighted according to the area which it represents
            (i.e. multiplying the weight by its range).
        cmap: colors to use for visualization.

    Metrics:

    - `bev_chamfer`: chamfer distance (mean point cloud distance), in radar
      range bins.
    - `bev_hausdorff`: modified hausdorff distance (median), in range bins.
    """

    def __init__(
        self, weight: float = 1.0, bce_weight: float = 0.9,
        range_weighted: bool = True, cmap: str = 'inferno'
    ) -> None:
        self.weight = weight
        self.loss = CombinedDiceBCE(
            bce_weight=bce_weight, range_weighted=range_weighted)
        self.chamfer = Chamfer2D(mode="chamfer", on_empty=128.0)
        self.hausdorff = Chamfer2D(mode="modhausdorff", on_empty=128.0)
        self.cmap = cmap

    def metrics(
        self, y_true: dict[str, Shaped[Tensor, "..."]],
        y_hat: dict[str, Shaped[Tensor, "..."]],
        reduce: bool = True, train: bool = True
    ) -> Metrics:
        """Get training metrics."""
        loss = self.loss(y_hat['bev'], y_true['bev'], reduce=reduce)
        if train:
            return Metrics(loss=self.weight * loss, metrics={"bev_loss": loss})
        else:
            chamfer = self.chamfer(
                y_hat['bev'] > 0, y_true['bev'], reduce=reduce)
            hausdorff = self.hausdorff(
                y_hat['bev'] > 0, y_true['bev'], reduce=reduce)
            return Metrics(loss=loss, metrics={
                "bev_loss": loss, "bev_chamfer": chamfer,
                "bev_hausdorff": hausdorff})

    def visualizations(
        self, y_true: dict[str, Shaped[Tensor, "..."]],
        y_hat: dict[str, Shaped[Tensor, "..."]]
    ) -> dict[str, Shaped[np.ndarray, "H W 3"]]:
        """Generate visualizations."""
        return {
            "bev": comparison_grid(
                polar2_to_bev(y_true['bev'], height=256),
                polar2_to_bev(sigmoid(y_hat['bev']), height=256),
                cmap=self.cmap, cols=8)}
