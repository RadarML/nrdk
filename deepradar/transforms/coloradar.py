"""Coloradar Transforms."""

import json
import os

import numpy as np
from beartype.typing import Any, Sequence
from einops import rearrange
from jaxtyping import Bool, Float, Float32, UInt, UInt16

from .base import Transform


class ColoradarMap2d(Transform):
    """2D azimuth-range lidar map.

    Configured to load coloradar point cloud data.

        Augmentations:

        - `range_scale`: random scaling applied to ranges to the sensor.
        - `azimuth_flip`: flip along the azimuth axis.

    Args:
        z_min, z_max: minimum and maximum z-levels, relative to the lidar,
            to include in the BEV map, in meters.
        bins: number of azimuth bins.
    """

    def __init__(
        self, path: str, z_min: float = -0.6, z_max: float = 0.9,
        bins: int = 512
    ) -> None:
        with open(os.path.join(path, "radar", "radar.json")) as f:
            meta = json.load(f)

        self.resolution: float = meta["range_resolution"]
        self.bins: int = meta["shape"][-1]
        self.z_min = z_min
        self.z_max = z_max
        self.az_bins = bins

    def __call__(
        self, data: Float[np.ndarray, "T N xyz"], aug: dict[str, Any] = {},
        idx: int = 0
    ) -> Bool[np.ndarray, "T Az Nr"]:

        # Doesn't support batches for now
        assert(data.shape[0] == 1)
        data = data[0]

        x, y, z = data.T
        r = np.linalg.norm(data[:, :2], axis=1)

        x = x[r > 0]
        y = y[r > 0]
        z = z[r > 0]
        r = r[r > 0]

        az = np.arctan2(y, x)
        el = np.arcsin(z / r)

        # Crop to ex
        r[(z > self.z_max) | (z < self.z_min)] = 0
        # Crop to forward (+x)
        mask = (
            (az > -np.pi / 2) & (az < np.pi / 2)
            & (np.abs(el) * 180 / np.pi < 20))
        az = az[mask]
        r = r[mask]

        if "range_scale" in aug:
            r = r * aug["range_scale"]
        if aug.get("azimuth_flip"):
            az = -az

        # Create map
        res = np.zeros((self.az_bins, self.bins), dtype=bool)
        r_bin = (r / self.resolution).astype(np.int16)
        az_bin = ((az + np.pi / 2) * self.az_bins / np.pi).astype(np.int16)

        mask2 = (r_bin > 0) & (r_bin < self.bins)
        r_bin = r_bin[mask2]
        az_bin = az_bin[mask2]

        res[az_bin, r_bin] = True
        return res[None, :, :]
