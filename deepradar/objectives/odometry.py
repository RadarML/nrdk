"""Odometry objective."""

import numpy as np
import torch
from beartype.typing import Any
from jaxtyping import Float, Shaped
from torch import Tensor

from .base import Metrics, Objective


class Velocity(Objective):
    """Radar -> relative velocity.

    As a loss function, we use a l2 distance loss, with an added epsilon for
    numerical stability::

        l(v*, v^) = sqrt(||v* - v^||_2^2 + eps)

    Note that the model does not output the velocity directly, but rather a
    speed (`y[0]`) and direction vector (`y[1:4]`). The speed is also squared
    to ensure that it is always positive::

        vel = y[0]**2 * y[1:4] / ||y[1:4]||_2

    Args:
        weight: objective weight
        speed_eps: minimum speed (in range bins) to calculate normalized
            metrics such as percent or angle error.
        eps_angle: epsilon for numerical stability in angle calculations.

    Metrics:

    - `vel_speed`: absolute difference in estimated speed, in m/s
    - `vel_speedp`: relative difference in estimated speed
    - `vel_angle`: angular difference between the actual and predicted
      velocity vector, in degrees
    """

    def __init__(
        self, weight: float = 1.0, speed_eps: float = 2.0,
        eps: float = 1.0, eps_angle: float = 1e-5
    ) -> None:
        self.weight = weight
        self.speed_eps = speed_eps
        self.eps = eps
        self.eps_angle = eps_angle

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

    def angle(
        self, v1: Float[Tensor, "batch d"], v2: Float[Tensor, "batch d"]
    ) -> Float[Tensor, "batch"]:
        """Angle between two orthonormal tensors of vectors, in degrees."""
        cosine = torch.sum(v1 * v2, dim=1)
        angle = torch.arccos(torch.clip(
            cosine, -1 + self.eps_angle, 1 - self.eps_angle))
        return torch.abs(angle) * (180 / torch.pi)

    def metrics(
        self, y_true: dict[str, Shaped[Tensor, "..."]],
        y_hat: dict[str, Shaped[Tensor, "..."]],
        reduce: bool = True, train: bool = True
    ) -> Metrics:
        """Get training metrics."""
        v_true = y_true['vel']
        v_hat, v_dir, s_hat = self.activation(y_hat['vel'])

        diff: Float[Tensor, "batch 3"] = v_hat - v_true
        loss = torch.sqrt(torch.sum(diff * diff, dim=1) + self.eps)
        if reduce:
            loss = torch.mean(loss)

        with torch.no_grad():
            speed_true = torch.linalg.norm(v_true, dim=1)
            metrics = {
                "vel_speed": torch.abs(s_hat - speed_true),
                "vel_speedp": torch.where(
                    speed_true > self.speed_eps,
                    torch.abs((s_hat - speed_true) / speed_true), 0.0),
                "vel_angle": torch.where(
                    speed_true > self.speed_eps,
                    self.angle(v_dir, v_true / speed_true[:, None]), 0.0)}

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
        y_hat: dict[str, Shaped[Tensor, "batch ..."]], gt: bool = True
    ) -> dict[str, Shaped[np.ndarray, "batch ..."]]:
        """Summarize predictions to visualize later.

        Args:
            y_true, y_hat: see :py:meth:`Objective.metrics`.
            gt: whether to render ground truth.

        Returns:
            A dict, where each key is the name of a visualization or output
            data, and the value is a quantized or packed format if possible.
        """
        v_hat, _, _, = self.activation(y_hat['vel'])
        res = {"vel": v_hat.to(torch.float32).cpu().numpy()}

        if gt:
            res["gt"] = y_true["vel"].to(torch.float32).cpu().numpy()

        return res
