"""Odometry objective."""

import torch
from jaxtyping import Float, Shaped
from torch import Tensor

from .base import Metrics, Objective


class Velocity(Objective):
    """Radar -> relative velocity.

    Args:
        weight: objective weight
        loss_order: loss type (l1/l2).

    Metrics:

    - `vel_speed`: absolute difference in estimated speed, in m/s
    - `vel_speedp`: relative difference in estimated speed
    - `vel_angle`: angular difference between the actual and predicted
      velocity vector, in degrees
    """

    def __init__(self, weight: float = 1.0, loss_order: int = 1) -> None:
        self.weight = weight
        self.loss_order = loss_order

    def metrics(
        self, y_true: dict[str, Shaped[Tensor, "..."]],
        y_hat: dict[str, Shaped[Tensor, "..."]],
        reduce: bool = True, train: bool = True
    ) -> Metrics:
        """Get training metrics."""
        diff: Float[Tensor, "batch 3"] = y_hat['vel']- y_true['vel']
        if self.loss_order == 1:
            diff = torch.abs(diff)
        elif self.loss_order == 2:
            diff = diff * diff
        else:
            raise NotImplementedError("Only L1 and L2 losses are implemented.")

        loss = torch.sum(diff, dim=1)
        if reduce:
            loss = torch.mean(loss)

        with torch.no_grad():
            speed_hat = torch.linalg.norm(y_hat['vel'], dim=1)
            speed_true = torch.linalg.norm(y_true['vel'], dim=1)
            cosine = torch.sum(
                y_hat['vel'] * y_true['vel'], dim=1
            ) / (speed_hat * speed_true)
            metrics = {
                "vel_speed": torch.abs(speed_hat - speed_true),
                "vel_speedp": torch.abs((speed_hat - speed_true) / speed_true),
                "vel_angle": torch.abs(torch.arccos(cosine) * 180 / torch.pi)}
            if reduce:
                metrics = {k: torch.nanmean(v) for k, v in metrics.items()}
            metrics["vel_loss"] = loss

        return Metrics(loss=self.weight * loss, metrics=metrics)
