"""2D Occupancy classification objective."""

import numpy as np
import torch
from beartype.typing import Any
from jaxtyping import Bool, Float, Shaped
from torch import Tensor
from torch.nn.functional import binary_cross_entropy_with_logits, sigmoid

from deepradar.utils import comparison_grid, polar2_to_bev

from .base import (
    Metrics,
    MetricValue,
    Objective,
    PointCloudObjective,
    accuracy_metrics,
)


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


class Chamfer2D(PointCloudObjective):
    """Chamfer metric for 2D polar occupancy grids.

    See :py:class:`.PointCloudObjective`.
    """

    @staticmethod
    def as_points(
        data: Bool[Tensor, "azimuth range"]
    ) -> Float[Tensor, "n 2"]:
        """Convert (azimuth, range) occupancy grid to points."""
        Na, Nr = data.shape
        bin, r = torch.where(data)
        phi = (bin - Na // 2) / Na * np.pi
        x = torch.cos(phi) * r
        y = torch.sin(phi) * r
        return torch.stack([x, y], dim=-1)


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
    - `bev_(tpr|fpr|tnr|fnr)`: true/false positive/negative rate.
    - `bev_(precision|recall)`: precision and recall.
    - `bev_acc`: accuracy (`tnr + tnr`).
    - `bev_f1`: F1 score (`2 precision * recall / (precision + recall)`).
    """

    def __init__(
        self, weight: float = 1.0, bce_weight: float = 0.9,
        range_weighted: bool = True, cmap: str = 'inferno'
    ) -> None:
        self.weight = weight
        self.loss = CombinedDiceBCE(
            bce_weight=bce_weight, range_weighted=range_weighted)
        self.chamfer = Chamfer2D(mode="chamfer", on_empty=128.0)
        self.cmap = cmap

    def metrics(
        self, y_true: dict[str, Shaped[Tensor, "..."]],
        y_hat: dict[str, Shaped[Tensor, "..."]],
        reduce: bool = True, train: bool = True
    ) -> Metrics:
        """Get training metrics."""
        # Canonical batch-elevation-azimuth-range -> batch-azimuth-range
        bev_hat = y_hat['bev'][:, 0]

        loss = self.loss(bev_hat, y_true['bev'], reduce=reduce)
        if train:
            return Metrics(loss=self.weight * loss, metrics={"bev_loss": loss})
        else:
            occ = bev_hat > 0
            chamfer = self.chamfer(occ, y_true['bev'], reduce=reduce)
            return Metrics(loss=loss, metrics={
                "bev_loss": loss,
                "bev_chamfer": chamfer,
                **accuracy_metrics(
                    occ, y_true['bev'], prefix="bev_", reduce=reduce)
            })

    def visualizations(
        self, y_true: dict[str, Shaped[Tensor, "..."]],
        y_hat: dict[str, Shaped[Tensor, "..."]]
    ) -> dict[str, Shaped[np.ndarray, "H W 3"]]:
        """Generate visualizations."""
        # Canonical batch-elevation-azimuth-range -> batch-azimuth-range
        bev_hat = y_hat['bev'][:, 0]

        return {
            "bev": comparison_grid(
                polar2_to_bev(y_true['bev'], height=256),
                polar2_to_bev(sigmoid(bev_hat), height=256),
                cmap=self.cmap, cols=8)}

    RENDER_CHANNELS: dict[str, dict[str, Any]] = {
        "bev": {
            "format": "lzma", "type": "u1", "shape": [256, 512],
            "desc": "BEV in cartesian coordinates and quantized to 0-255."},
        "bev_gt": {
            "format": "lzma", "type": "u1", "shape": [256, 512],
            "desc": "Ground truth BEV view."}
    }

    def render(
        self, y_true: dict[str, Shaped[Tensor, "batch ..."]],
        y_hat: dict[str, Shaped[Tensor, "batch ..."]]
    ) -> dict[str, Shaped[np.ndarray, "batch ..."]]:
        """Summarize predictions to visualize later.

        Args:
            y_true, y_hat: see :py:meth:`Objective.metrics`.

        Returns:
            A dict, where each key is the name of a visualization or output
            data, and the value is a quantized or packed format if possible.
        """
        bev_hat = y_hat['bev'][:, 0]

        bev = polar2_to_bev(sigmoid(bev_hat), height=256) * 255
        bev_gt = polar2_to_bev(y_true['bev'], height=256) * 255
        return {
            "bev": bev.to(torch.uint8).cpu().numpy(),
            "bev_gt": bev_gt.to(torch.uint8).cpu().numpy()
        }
