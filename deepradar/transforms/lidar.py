"""Lidar transforms."""

import os, json
import numpy as np

from jaxtyping import UInt, UInt16, Float32, Bool
from beartype.typing import Any

from ouster import client

from .base import BaseTransform


class Destagger(BaseTransform):
    """Destagger lidar data.

    Augmentation parameters:

    - `range_scale`: random scaling applied to ranges to the sensor.
    - `azimuth_flip`: flip along the azimuth axis.
    """

    def __init__(self, path: str) -> None:
        # ouster-sdk is a naughty, noisy library
        # it is in fact so noisy, that we have cut it off at the os level...
        stdout = os.dup(1)
        os.close(1)
        with open(os.path.join(path, "lidar", "lidar.json")) as f:
            self.metadata = client.SensorInfo(f.read())
        os.dup2(stdout, 1)
        os.close(stdout)

    def __call__(
        self, data: UInt[np.ndarray, "El Az"], aug: dict[str, Any] = {}
    ) -> UInt[np.ndarray, "El Az"]:
        res = client.destagger(self.metadata, data)

        res *= aug.get("range_scale", 1.0)
        if aug.get("azimuth_flip"):
            res = np.flip(res, axis=1)

        return res


class Map2D(BaseTransform):
    """2D azimuth-range lidar map.

    Follows the procedure used by RadarHD [R1]_:

    1. Crop to only forward-facing regions; convert to polar coordinates.
    2. Discard points more than 30cm away from the radar plane.
    3. Write points to a 2D polar-coordinate occupancy grid. The range bins
       used in the map match the range bins measured by the radar.
    """

    def __init__(self, path: str) -> None:
        with open(os.path.join(path, "radar", "radar.json")) as f:
            self.resolution: float = json.load(f)["range_resolution"]
        with open(os.path.join(path, "radar", "meta.json")) as f:
            self.bins: int = json.load(f)["iq"]["shape"][-1] // 2

    def __call__(
        self, data: UInt16[np.ndarray, "El Az"], aug: dict[str, Any] = {}
    ) -> Bool[np.ndarray, "Az2 Nr"]:
        # Crop, convert mm -> m
        el, az = data.shape
        crop_el = el // 4
        crop_az = az // 4
        x_crop = data[crop_el:-crop_el, crop_az:-crop_az] / 1000

        # Project to polar
        el_angles = np.linspace(
            -np.pi / 2, np.pi / 2, x_crop.shape[0], dtype=np.float32)
        z = np.sin(el_angles)[:, None] * x_crop
        r = np.cos(el_angles)[:, None] * x_crop

        # Crop to (-30cm, 0cm)
        r[(z > 0.0) | (z < -0.3)] = 0

        # Create map
        bin = (r // (self.resolution)).astype(np.uint16)
        bin[bin >= self.bins] = 0
        res = np.zeros((bin.shape[1], self.bins), dtype=bool)
        for i in range(bin.shape[1]):
            res[i][bin[:, i]] = True
        res[:, 0] = 0
        return res


class DecimateMap(BaseTransform):
    """Downsample lidar map."""

    def __init__(self, path: str, azimuth: int = 1, range: int = 1) -> None:
        self.azimuth = azimuth
        self.range = range

    def __call__(
        self, data: Bool[np.ndarray, "Az Nr"]
    ) -> Bool[np.ndarray, "Az_dec Nr_dec"]:
        na, nr = data.shape
        return np.any(data.reshape(
            na // self.azimuth, self.azimuth, nr // self.range, self.range
        ), axis=(1, 3))


class Depth(BaseTransform):
    """Cropped depth map, in meters."""

    def __call__(
        self, data: UInt16[np.ndarray, "El Az"], aug: dict[str, Any] = {}
    ) -> Float32[np.ndarray, "El2 Az2"]:
        el, az = data.shape
        crop_el = el // 4
        crop_az = az // 4
        x_crop = data[crop_el:-crop_el, crop_az:-crop_az]
        return x_crop.astype(np.float32) / 1000
