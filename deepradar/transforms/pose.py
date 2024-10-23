"""Pose/odometry transforms."""

import json
import os

import numpy as np
from beartype.typing import Any
from jaxtyping import Float

from .base import Transform


class RelativeVelocity(Transform):
    """Velocity relative to the sensor's frame of reference.

    Implementation Notes:

        The relative velocity of the rig (after transforming from world space
        to sensor space via `inv(rot)`) is specified using a FLU convention;
        when viewed relative to an operator standing behind the rig, `+x`
        points forward, `+y` to the left, and `+z` up. These components are
        then indexed conventionally (in `xyz` order).

    Augmentations:

        - `speed_scale`: multiply velocity by speed scale.
        - `doppler_flip`: reverse the relative velocity.
        - `azimuth_flip`: reverse the `y` component of the velocity.
    """

    def __init__(self, path: str) -> None:
        with open(os.path.join(path, "radar", "radar.json")) as f:
            self.resolution: float = json.load(f)["doppler_resolution"]

    def __call__(
        self, data: dict[str, Float[np.ndarray, "..."]],
        aug: dict[str, Any] = {}, idx: int = 0
    ) -> Float[np.ndarray, "3"]:
        scale = aug.get("speed_scale", 1.0) / self.resolution
        vel = np.matmul(np.linalg.inv(data["rot"]), data["vel"]) * scale

        if aug.get("doppler_flip"):
            vel *= -1
        if aug.get("azimuth_flip"):
            vel[1] = vel[1] * -1

        return vel
