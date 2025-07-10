"""Odometry objective."""

from typing import Protocol, runtime_checkable

import numpy as np
import torch
from abstract_dataloader.ext.objective import Objective
from einops import reduce
from jaxtyping import Float, Shaped
from torch import Tensor


@runtime_checkable
class VelocityData(Protocol):
    """Protocol type for velocity data.

    Attributes:
        vel: velocity vector, with batch-3 axis order.
    """

    vel: Float[Tensor, "batch t 3"]


class Velocity(Objective[Tensor, VelocityData, Float[Tensor, "batch t 4"]]):
    """Radar -> relative velocity.

    !!! info

        As a loss function, we use a l2 distance loss, with an added epsilon
        for numerical stability:
        ```
        l(v*, v^) = sqrt(||v* - v^||_2^2 + eps)
        ```
        Note that the model does not output the velocity directly, but rather a
        speed (`y[0]`) and direction vector (`y[1:]`). The speed is also
        squared to ensure that it is always positive:
        ```
        vel = y[0]**2 * y[1:] / ||y[1:]||_2
        ```

    Metrics:
        - `vel_speed`: absolute difference in estimated speed, in m/s
        - `vel_speedp`: relative difference in estimated speed
        - `vel_angle`: angular difference between the actual and predicted
        velocity vector, in degrees

    Args:
        eps: epsilon for numerical stability in sqrt, log, etc.
        eps_speed: minimum speed (in range bins) to calculate normalized
            metrics such as percent or angle error.
        eps_angle: epsilon for numerical stability in angle calculations.
    """

    def __init__(
        self, eps: float = 1.0, eps_speed: float = 2.0, eps_angle: float = 1e-5
    ) -> None:
        self.eps_speed = eps_speed
        self.eps = eps
        self.eps_angle = eps_angle

    @staticmethod
    def _activation(
        outputs: Float[Tensor, "b 4"]
    ) -> tuple[Float[Tensor, "b 3"], Float[Tensor, "b 3"], Float[Tensor, "b"]]:
        """Get velocity and speed from model outputs."""
        s_raw = outputs[:, 0]
        v_raw = outputs[:, 1:]

        v_dir = v_raw / torch.linalg.norm(v_raw, dim=1)[:, None]
        s_hat = s_raw**2
        v_hat = s_hat[:, None] * v_dir

        return v_hat, v_dir, s_hat

    def _angle(
        self, v1: Float[Tensor, "batch d"], v2: Float[Tensor, "batch d"]
    ) -> Float[Tensor, "batch"]:
        """Angle between two orthonormal tensors of vectors, in degrees."""
        cosine = torch.sum(v1 * v2, dim=1)
        angle = torch.arccos(torch.clip(
            cosine, -1 + self.eps_angle, 1 - self.eps_angle))
        return torch.abs(angle) * (180 / torch.pi)

    def __call__(
        self, y_true: VelocityData, y_pred: Float[Tensor, "batch t 4"],
        train: bool = True
    ) -> tuple[Tensor, dict[str, Float[Tensor, "batch"]]]:
        v_true = y_true.vel.reshape(-1, 3)
        v_hat, v_dir, s_hat = self._activation(y_pred.reshape(-1, 4))

        with torch.no_grad():
            speed_true = torch.linalg.norm(v_true, dim=1)
            metrics = {
                "speed": torch.abs(s_hat - speed_true),
                "speedp": torch.where(
                    speed_true > self.eps_speed,
                    torch.abs((s_hat - speed_true) / speed_true), 0.0),
                "angle": torch.where(
                    speed_true > self.eps_speed,
                    self._angle(v_dir, v_true / speed_true[:, None]), 0.0)}

        diff: Float[Tensor, "batch 3"] = v_hat - v_true
        metrics["loss"] = torch.sqrt(torch.sum(diff * diff, dim=1) + self.eps)

        # Temporal reduction
        metrics = {
            k: reduce(
                v, "(b t) -> b", "mean", b=y_pred.shape[0], t=y_pred.shape[1])
            for k, v in metrics.items()}
        return metrics["loss"], metrics

    def visualizations(
        self, y_true: VelocityData, y_pred: Float[Tensor, "batch t 4"]
    ) -> dict[str, Shaped[np.ndarray, "H W 3"]]:
        return {}

    def render(
        self, y_true: VelocityData,
        y_pred: Float[Tensor, "batch t 4"], render_gt: bool = True
    ) -> dict[str, Shaped[np.ndarray, "batch ..."]]:
        v_hat, _, _, = self._activation(y_pred)
        res = {"vel": v_hat.to(torch.float32).cpu().numpy()}
        if render_gt:
            res["vel_gt"] = y_true.vel.to(torch.float32).cpu().numpy()

        return res
