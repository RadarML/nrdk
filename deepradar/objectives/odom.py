"""Odometry objective."""

import torch
from torch import Tensor
from jaxtyping import Float, Shaped

from .base import MetricValue, Metrics, Objective


class LPLoss:
    """Generic lp loss.

    Args:
        ord: loss order; only l1 (`1`) and l2 (`2`) are supported.
    """

    def __init__(self, ord: int = 1) -> None:
        self.ord = ord

    def __call__(
        self, y_hat: Float[Tensor, "batch ..."],
        y_true: Float[Tensor, "batch ..."], reduce: bool = True
    ) -> MetricValue:
        if self.ord == 1:
            res = torch.abs(y_hat - y_true)
        elif self.ord == 2:
            res = (y_hat - y_true)**2
        else:
            raise NotImplementedError("Only L1 and L2 losses are implemented.")

        while len(res.shape) > 2:
            res = torch.sum(res, dim=-1)

        if reduce:
            return torch.mean(res)
        else:
            return res


class Velocity(Objective):
    """Radar -> relative velocity.

    Args:
        weight: objective weight
        loss_order: loss type (l1/l2).
    """

    def __init__(self, weight: float = 1.0, loss_order: int = 1) -> None:
        self.weight = weight
        self.loss = LPLoss(ord=loss_order)

    def metrics(
        self, y_true: dict[str, Shaped[Tensor, "..."]],
        y_hat: dict[str, Shaped[Tensor, "..."]],
        reduce: bool = True, train: bool = True
    ) -> Metrics:
        """Get training metrics."""
        loss = self.loss(y_hat['vel'], y_true['vel'], reduce=reduce)
        return Metrics(loss=self.weight * loss, metrics={"vel_loss": loss})
