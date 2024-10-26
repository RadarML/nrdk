"""Odometry objective."""

import numpy as np
import torch
from beartype.typing import Any
from jaxtyping import Float, Shaped
from torch import Tensor

from .base import Metrics, Objective


class Velocity(Objective):
    """Radar -> relative velocity.

    Note that the model does not output the velocity directly, but rather a
    speed (`y[0]`) and direction vector (`y[1:4]`). The speed is also squared
    to ensure that it is always positive::

        vel = y[0]**2 * y[1:4] / ||y[1:4]||_2

    Args:
        weight: objective weight
        loss_order: loss type (l1/l2).
        speed_eps: minimum speed (in range bins) to calculate normalized
            metrics such as percent or angle error.

    Metrics:

    - `vel_speed`: absolute difference in estimated speed, in m/s
    - `vel_speedp`: relative difference in estimated speed
    - `vel_angle`: angular difference between the actual and predicted
      velocity vector, in degrees
    """

    def __init__(
        self, weight: float = 1.0, loss_order: int = 1,
        speed_eps: float = 2.0
    ) -> None:
        self.weight = weight
        self.loss_order = loss_order
        self.speed_eps = speed_eps

    @staticmethod
    def activation(
        outputs: Float[Tensor, "b 4"]
    ) -> tuple[Float[Tensor, "b 3"], Float[Tensor, "b 3"], Float[Tensor, "b"]]:
        """Get velocity and speed from model outputs."""
        s_raw = outputs[:, 0]
        v_raw = outputs[:, 1:]

        v_dir = v_raw / torch.linalg.norm(v_raw, dim=1)[:, None]
        s_hat = s_raw**2
        v_hat = s_hat[:, None] * v_dir

        return v_hat, v_dir, s_hat

    def metrics(
        self, y_true: dict[str, Shaped[Tensor, "..."]],
        y_hat: dict[str, Shaped[Tensor, "..."]],
        reduce: bool = True, train: bool = True
    ) -> Metrics:
        """Get training metrics."""
        v_hat, v_dir, s_hat = self.activation(y_hat['vel'])

        diff: Float[Tensor, "batch 3"] = v_hat - y_true['vel']
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
            speed_true = torch.linalg.norm(y_true['vel'], dim=1)
            cosine = torch.sum(v_dir * y_true['vel'], dim=1) / speed_true
            metrics = {
                "vel_speed": torch.abs(s_hat - speed_true),
                "vel_speedp": torch.where(
                    speed_true > self.speed_eps,
                    torch.abs((s_hat - speed_true) / speed_true), 0.0),
                "vel_angle": torch.where(
                    speed_true > self.speed_eps,
                    torch.abs(torch.arccos(cosine) * 180 / torch.pi), 0.0)}

            if reduce:
                metrics = {k: torch.nanmean(v) for k, v in metrics.items()}
            metrics["vel_loss"] = loss

        return Metrics(loss=self.weight * loss, metrics=metrics)

    RENDER_CHANNELS: dict[str, dict[str, Any]] = {
        "vel": {
            "format": "raw", "type": "f4", "shape": [3],
            "desc": "Ego-velocity vector."},
        "vel_gt": {
            "format": "raw", "type": "f4", "shape": [3],
            "desc": "Ground truth ego-velocity."}
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
        v_hat, _, _, = self.activation(y_hat['vel'])
        return {
            "vel": v_hat.to(torch.float32).cpu().numpy(),
            "vel_gt": y_true["vel"].to(torch.float32).cpu().numpy()
        }
