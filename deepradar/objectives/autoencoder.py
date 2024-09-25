"""Radar autoencoder objective."""

import numpy as np
import torch
from beartype.typing import Literal
from jaxtyping import Float, Shaped
from torch import Tensor
from torchvision.transforms import InterpolationMode, Resize

from deepradar.utils import comparison_grid

from .base import LPObjective, Metrics, Objective


class Radar(Objective):
    """Radar -> radar autoencoder objective.

    Args:
        weight: objective weight.
        loss_order: Loss type (l1/l2).
        cmap: colors to use for visualizations.
    """

    def __init__(
        self, weight: float = 1.0, loss_order: Literal[1, 2] = 1,
        cmap: str = 'viridis'
    ) -> None:
        self.weight = weight
        self.loss = LPObjective(loss_order)
        self.cmap = cmap

    def metrics(
        self, y_true: dict[str, Shaped[Tensor, "..."]],
        y_hat: dict[str, Shaped[Tensor, "..."]],
        reduce: bool = True, train: bool = True
    ) -> Metrics:
        """Get training metrics."""
        loss = self.loss(y_hat['radar'], y_true['radar'], reduce=reduce)
        return Metrics(loss=self.weight * loss, metrics={"radar_loss": loss})

    def visualizations(
        self, y_true: dict[str, Shaped[Tensor, "..."]],
        y_hat: dict[str, Shaped[Tensor, "..."]]
    ) -> dict[str, Shaped[np.ndarray, "H W 3"]]:
        """Generate visualizations."""
        rez = Resize((256, 256 * 2), interpolation=InterpolationMode.NEAREST)
        return {"depth": comparison_grid(
            rez(y_true['depth']), rez(y_hat['depth']), cmap=self.cmap, cols=8)}
