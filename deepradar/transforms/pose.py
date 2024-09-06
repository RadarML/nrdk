"""Pose/odometry transforms."""

import json
import os

import numpy as np
from beartype.typing import Any
from jaxtyping import Float

from .base import Transform


class RelativeVelocity(Transform):
    """Velocity relative to the sensor's frame of reference."""

    def __init__(self, path: str) -> None:
        with open(os.path.join(path, "radar", "radar.json")) as f:
            self.resolution: float = json.load(f)["doppler_resolution"]

    def __call__(
        self, data: dict[str, Float[np.ndarray, "..."]],
        aug: dict[str, Any] = {}
    ) -> Float[np.ndarray, "3"]:
        scale = aug.get("speed_scale", 1.0)
        vel = np.matmul(np.linalg.inv(data["rot"]), data["vel"]) * scale

        if aug.get("doppler_flip"):
            vel *= -1
        if aug.get("azimuth_flip"):
            vel[1] = vel[1] * -1

        return vel
