"""3D depth estimation objective."""

import numpy as np
import torch
from jaxtyping import Float, Shaped
from torch import Tensor
from torchvision.transforms import InterpolationMode, Resize

from deepradar.utils import comparison_grid

from .base import Metrics, MetricValue, Objective


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
            diff = (y_hat - y_true)**2
        else:
            raise NotImplementedError("Only L1 and L2 losses are implemented.")

        mask = (y_true != 0)
        res = torch.sum(diff * mask, dim=(1, 2)) / torch.sum(mask, dim=(1, 2))
        if reduce:
            res = torch.mean(res)
        return res


class Depth(Objective):
    """Radar -> lidar as depth estimation.

    NOTE: not up to date!

    Missing values (`range==0`) are ignored during loss calculation, and
    treated as "void".

    Args:
        weight: objective weight.
        loss_order: Loss type (l1/l2).
        cmap: colors to use for visualizations.
    """

    def __init__(
        self, weight: float = 1.0, loss_order: int = 1, cmap: str = 'viridis'
    ) -> None:
        self.weight = weight
        self.loss = LPDepth(ord=loss_order)
        self.cmap = cmap

    def metrics(
        self, y_true: dict[str, Shaped[Tensor, "..."]],
        y_hat: dict[str, Shaped[Tensor, "..."]],
        reduce: bool = True, train: bool = True
    ) -> Metrics:
        """Get training metrics."""
        # Canonical elevation-azimuth-range -> elevation-azimuth
        depth_hat = y_hat["depth"][..., 0]

        loss = self.loss(depth_hat, y_true['depth'], reduce=reduce)
        return Metrics(loss=self.weight * loss, metrics={"depth_loss": loss})

    def visualizations(
        self, y_true: dict[str, Shaped[Tensor, "..."]],
        y_hat: dict[str, Shaped[Tensor, "..."]], gt: bool = True
    ) -> dict[str, Shaped[np.ndarray, "H W 3"]]:
        """Generate visualizations."""
        # Canonical elevation-azimuth-range -> elevation-azimuth
        depth_hat = y_hat["depth"][..., 0]

        rez = Resize((256, 256 * 2), interpolation=InterpolationMode.NEAREST)
        return {"depth": comparison_grid(
            rez(y_true['depth']), rez(depth_hat), cmap=self.cmap, cols=8)}
