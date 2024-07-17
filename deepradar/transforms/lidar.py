"""Lidar transforms."""

import os, json
import numpy as np

from jaxtyping import UInt, UInt16, Float32, Bool
from beartype.typing import Any

from ouster import client

from .base import BaseTransform


class Destagger(BaseTransform):
    """Destagger lidar data.

    Augmentations:

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

        if "range_scale" in aug:
            res = (
                res.astype(np.float32) * aug["range_scale"]).astype(np.uint16)
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
            meta = json.load(f)
        self.resolution: float = meta["range_resolution"]
        self.bins: int = meta["shape"][-1]

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
        res[:, 0] = False
        return res


class DecimateMap(BaseTransform):
    """Downsample lidar map.
    
    Args:
        azimuth, range: azimuth and range decimation factors.
    """

    def __init__(self, path: str, azimuth: int = 1, range: int = 1) -> None:
        self.azimuth = azimuth
        self.range = range

    def __call__(
        self, data: Bool[np.ndarray, "Az Nr"], aug: dict[str, Any] = {}
    ) -> Bool[np.ndarray, "Az_dec Nr_dec"]:
        na, nr = data.shape
        return np.any(data.reshape(
            na // self.azimuth, self.azimuth, nr // self.range, self.range
        ), axis=(1, 3))


class Depth(BaseTransform):
    """Cropped depth map, in meters.

    Implementation notes:

    - The depth is automatically cropped to the maximum range of the radar as
      recorded in `radar/radar.json` by setting all values exceeding the max
      range to 0 (e.g. not known).
    - The max depth is normalized to have value
      1.0.
    - Elevation and azimuth crop factors are applied symetrically, e.g.
      `crop_el = 0.25` means removing `0.25` of the elevation bins from the top
      and bottom, leaving only half of the measured bins.
    - The Ouster lidar has a 360 degree azimuth FOV. Since the radar can only
      see forward, `crop_az` should be at least `0.25`.

    Args:
        path: path to dataset.
        crop_el, crop_az: crop factor to apply (symmetrically) to each side
    """
        
    def __init__(
        self, path: str, crop_el: float = 0.0, crop_az: float = 0.25
    ) -> None:
        with open(os.path.join(path, "radar", "radar.json")) as f:
            meta = json.load(f)
        self.max_range: float = meta["range_resolution"] * meta["shape"][-1]
        self.crop_el = crop_el
        self.crop_az = crop_az

    def __call__(
        self, data: UInt16[np.ndarray, "El Az"], aug: dict[str, Any] = {}
    ) -> Float32[np.ndarray, "El2 Az2"]:
        el, az = data.shape
        crop_el = int(el * self.crop_el)
        crop_az = int(az * self.crop_az)

        if crop_el > 0:
            data = data[crop_el:-crop_el]
        if crop_az > 0:
            data = data[:, crop_az:-crop_az]

        # Note: the Ouster lidar natively uses mm as a unit, while we use m.
        depth_m = data.astype(np.float32) / 1000
        depth_m[depth_m > self.max_range] = 0.0
        return depth_m / self.max_range
