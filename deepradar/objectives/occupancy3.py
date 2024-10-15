"""3D Occupancy classification objective."""

import numpy as np
import torch
from jaxtyping import Shaped
from torch import Tensor
from torch.nn.functional import binary_cross_entropy_with_logits

from deepradar.utils import comparison_grid, polar3_to_bev

from .base import Metrics, Objective


class PolarOccupancy(Objective):
    """Radar -> lidar as 3D polar occupancy.

    Args:
        weight: total objective weight.
        range_weighted: Whether to apply equal-area "range weighting", where
            the relative weight of bins is adjusted on the range axis so that
            each bin is weighted according to the area which it represents
            (i.e. multiplying the weight by `range**2`).
        positive_weight: weight to give occupied samples; nominally the number
            of range bins.
        cmap_bev, cmap_depth: colors to use for visualization.

    Metrics:

    - `bev_chamfer`: chamfer distance (mean point cloud distance), in radar
      range bins.
    - `bev_hausdorff`: modified hausdorff distance (median), in range bins.
    """

    def __init__(
        self, weight: float = 1.0, range_weighted: bool = True,
        positive_weight: float = 64.0,
        cmap_bev: str = 'inferno', cmap_depth: str = 'viridis'
    ) -> None:
        self.weight = weight
        self.range_weighted = range_weighted
        self.positive_weight = positive_weight
        self.cmap_bev = cmap_bev
        self.cmap_depth = cmap_depth

    def metrics(
        self, y_true: dict[str, Shaped[Tensor, "..."]],
        y_hat: dict[str, Shaped[Tensor, "..."]],
        reduce: bool = True, train: bool = True
    ) -> Metrics:
        """Get training metrics."""
        # Range weighting
        batch, ne, na, nr = y_true['map'].shape
        if self.range_weighted:
            bins = torch.arange(nr, device=y_true['map'].device)
            weight = ((bins + 1) / nr)[None, None, None, :] ** 2
        else:
            # Mypy doesn't seem to recognize the type overloading here.
            weight = 1.0  # type: ignore

        # Class weighting
        weight = torch.where(y_true['map'], self.positive_weight, 1.0) * weight
        total_weight = torch.sum(weight, dim=(1, 2, 3))

        loss = weight * binary_cross_entropy_with_logits(
            y_hat['map'], y_true['map'].to(y_hat['map'].dtype),
            reduction='none')
        loss = torch.sum(loss, dim=(1, 2, 3)) / total_weight

        if reduce:
            loss = torch.sum(loss)

        return Metrics(loss=self.weight * loss, metrics={"map_loss": loss})

    def visualizations(
        self, y_true: dict[str, Shaped[Tensor, "..."]],
        y_hat: dict[str, Shaped[Tensor, "..."]]
    ) -> dict[str, Shaped[np.ndarray, "H W 3"]]:
        """Generate visualizations."""
        depth = torch.argmax(y_true['map'].to(torch.uint8), dim=-1)
        bev = polar3_to_bev(y_true['map'], mode='highest')
        bev[bev == 0] = torch.min(bev) - 2

        map_pred = y_hat['map'] > 0
        depth_hat = torch.argmax(map_pred.to(torch.uint8), dim=-1)
        bev_hat = polar3_to_bev(map_pred, mode="highest")
        bev_hat[bev_hat == 0] = torch.min(bev_hat) - 2

        return {
            "bev": comparison_grid(
                bev, bev_hat, cmap=self.cmap_bev, cols=8, normalize=True),
            "depth": comparison_grid(
                depth, depth_hat, cmap=self.cmap_depth, cols=8,
                normalize=True)}
