"""Lidar transforms."""

import json
import os

import numpy as np
from beartype.typing import Any, Sequence
from einops import rearrange
from jaxtyping import Bool, Float, Float32, UInt, UInt16

# Ouster imports are broken for type checking as of 0.12.0, so we have to
# ignore type checking any time we use anything...
from ouster.sdk import client

from .base import Transform


class Destagger(Transform):
    """Destagger lidar data.

    NOTE: while this acts as a "root" transform for lidar data, and could be
    used to apply data augmentations, we instead apply augmentations later so
    that they only need to be calculated on post-cropped data.
    """

    def __init__(self, path: str) -> None:
        # ouster-sdk is a naughty, noisy library
        # it is in fact so noisy, that we have cut it off at the os level...
        stdout = os.dup(1)
        os.close(1)
        with open(os.path.join(path, "lidar", "lidar.json")) as f:
            self.metadata = client.SensorInfo(f.read())  # type: ignore
        os.dup2(stdout, 1)
        os.close(stdout)

    def __call__(
        self, data: UInt[np.ndarray, "T El Az"], aug: dict[str, Any] = {},
        idx: int = 0
    ) -> UInt[np.ndarray, "T El Az"]:
        """Destagger data.

        Args:
            data: input frames; note that the axis order is in memory order,
                which is opposite to ouster order.

        Returns:
            Destaggered data, rearranged back to memory order.
        """
        raw_os = rearrange(data, "t el az -> el az t")
        out_os = client.destagger(self.metadata, raw_os)  # type: ignore
        return rearrange(out_os, "el az t -> t el az")

class Map2D(Transform):
    """2D azimuth-range lidar map.

    Follows the procedure used by RadarHD [T1]_:

    1. Crop to only forward-facing regions; convert to polar coordinates.
    2. Discard points more than 30cm away from the radar plane.
    3. Write points to a 2D polar-coordinate occupancy grid. The range bins
       used in the map match the range bins measured by the radar.

    Implementation Notes:

        `Map2D` excludes the ground (and ceiling) using a z-level clip
        specified by `z_min, z_max`, with the z coordinate being relative to
        the lidar along the vertical axis of the lidar sensor.

        - The radar is at approximately `z=-0.15`, so `z_min, z_max` are not
          necessarily centered.
        - `z_min` should be set to less than the distance between the lidar
          and the ground; `z_max` should be set to less than the distance
          between the lidar and the lowest ceiling expected to be encountered.
        - Be cautious of an overly tight `z_max - z_min`: the OS0-128, for
          example, only has `90 / 128 ~ 0.70` degrees of angular resolution. At
          range `D`, only `~ (z_max - z_min) / (D sin(0.70 deg))` azimuth bins
          will fall in this region. For `z_max - z_min = 0.6` and `D = 25m`,
          this corresponds to only ~2 bins!
        - Note that the BEV transform here does not account for the orientation
          of the rig; if the rig is tilted, the BEV plane tilts as well.

    Augmentations:

        - `range_scale`: random scaling applied to ranges to the sensor.
        - `azimuth_flip`: flip along the azimuth axis.

    Args:
        z_min, z_max: minimum and maximum z-levels, relative to the lidar,
            to include in the BEV map, in meters.
        crop_el, crop_az: crop factor to apply (symmetrically) to each side;
            these values are excluded from BEV calculation. `crop_az` should
            always be `0.25`; `crop_el` can be adjusted to crop nearby objects
            more to avoid spread due to angled objects.
    """

    def __init__(
        self, path: str, z_min: float = -0.6, z_max: float = 0.9,
        crop_el: float = 0.25, crop_az: float = 0.25
    ) -> None:
        with open(os.path.join(path, "radar", "radar.json")) as f:
            meta = json.load(f)
        with open(os.path.join(path, "lidar", "lidar.json")) as f:
            intrinsics = json.load(f)["beam_intrinsics"]
            self.beam_angles: Float[np.ndarray, "beams"] = np.array(
                intrinsics["beam_altitude_angles"]
            ) / 180 * np.pi

        self.resolution: float = meta["range_resolution"]
        self.bins: int = meta["shape"][-1]
        self.z_min = z_min
        self.z_max = z_max
        self.crop_el = crop_el
        self.crop_az = crop_az

    def __call__(
        self, data: UInt16[np.ndarray, "T El Az"], aug: dict[str, Any] = {},
        idx: int = 0
    ) -> Bool[np.ndarray, "T Az2 Nr"]:
        # Crop, convert mm -> m
        t, el, az = data.shape
        crop_el = int(self.crop_el * el)
        crop_az = int(self.crop_az * az)
        x_crop = data[:, crop_el:-crop_el, crop_az:-crop_az] / 1000

        # Project to polar
        angles = self.beam_angles[crop_el:-crop_el]
        z = np.sin(angles)[None, :, None] * x_crop
        r = np.cos(angles)[None, :, None] * x_crop

        # Crop to ex
        r[(z > self.z_max) | (z < self.z_min)] = 0

        if "range_scale" in aug:
            r = r * aug["range_scale"]
        if aug.get("azimuth_flip"):
            r = np.flip(r, axis=2)

        # Create map
        bin: UInt16[np.ndarray, "T El Az"] = (
            r // (self.resolution)).astype(np.uint16)
        bin[bin >= self.bins] = 0


        t, el2, az2 = bin.shape
        res = np.zeros((t, az2, self.bins), dtype=bool)
        i_t, i_az = np.meshgrid(np.arange(t), np.arange(az2), indexing='ij')
        res[i_t[:, :, None], i_az[:, :, None], bin.transpose(0, 2, 1)] = True
        res[:, :, 0] = False
        return res


class Map3D(Transform):
    """3D elevation-azimuth-range lidar map.

    Augmentations:

        - `range_scale`: random scaling applied to ranges to the sensor.
        - `azimuth_flip`: flip along the azimuth axis.

    Args:
        decimate: (elevation, elevation, range) decimation to apply.
        crop_az: amount of (symmetric) cropping to apply in the azimuth axis.
            Used to eliminate rear-facing lidar points.
    """

    def __init__(
        self, path: str, decimate: Sequence[int] = (1, 1, 1),
        crop_az: float = 0.25
    ) -> None:
        if len(decimate) != 3:
            raise ValueError("Must have a 3D decimation factor.")

        with open(os.path.join(path, "radar", "radar.json")) as f:
            meta = json.load(f)

        d_el, d_az, d_rng = decimate
        self.resolution: float = meta["range_resolution"] * d_rng
        self.bins: int = meta["shape"][-1] // d_rng
        self.d_el = d_el
        self.d_az = d_az

        self.crop_az = crop_az

    def __call__(
        self, data: UInt16[np.ndarray, "T El Az"], aug: dict[str, Any] = {},
        idx: int = 0
    ) -> Bool[np.ndarray, "T El2 Az2 Nr"]:
        # Crop, convert mm -> m
        t, el, az = data.shape
        crop_az = int(self.crop_az * az)
        rng = data[:, :, crop_az:-crop_az] / 1000

        if "range_scale" in aug:
            rng = rng * aug["range_scale"]
        if aug.get("azimuth_flip"):
            rng = np.flip(rng, axis=1)

        # Create map
        bin = (rng // (self.resolution)).astype(np.uint16)
        bin[bin >= self.bins] = 0

        t, el2, az2 = bin.shape
        n_el = el2 // self.d_el
        n_az = az2 // self.d_az

        bin = rearrange(
            bin, "t (el d_el) (az d_az) -> t el d_el az d_az",
            el=n_el, d_el=self.d_el, az=n_az, d_az=self.d_az)
        res = np.zeros((t, n_el, n_az, self.bins), dtype=bool)

        i_t = np.broadcast_to(
            np.arange(t)[:, None, None, None, None], bin.shape).flatten()
        i_el = np.broadcast_to(
            np.arange(n_el)[None, :, None, None, None], bin.shape).flatten()
        i_az = np.broadcast_to(
            np.arange(n_az)[None, None, None, :, None], bin.shape).flatten()
        res[i_t, i_el, i_az, bin.flatten()] = True
        res[:, :, :, 0] = False

        return res


class Depth(Transform):
    """Cropped depth map, in meters.

    Pixels with no return (e.g. infinite depth or specular surface) are
    represented by `0.0`.

    Implementation notes:

        - The depth is automatically cropped to the maximum range of the radar
          as recorded in `radar/radar.json` by setting all values exceeding the
          max range to 0 (i.e. not known).
        - The max depth is normalized to have value 1.0.
        - Elevation and azimuth crop factors are applied symetrically, e.g.
          `crop_el = 0.25` means removing `0.25` of the elevation bins from the
          top and bottom, leaving only half of the measured bins.
        - The Ouster lidar has a 360 degree azimuth FOV. Since the radar can
          only see forward, `crop_az` should be at least `0.25`.

    Augmentations:

        - `range_scale`: random scaling applied to ranges to the sensor.
        - `azimuth_flip`: flip along the azimuth axis.

    Args:
        path: path to dataset.
        crop_el, crop_az: crop factor to apply (symmetrically) to each side.
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
        self, data: UInt16[np.ndarray, "T El Az"], aug: dict[str, Any] = {},
        idx: int = 0
    ) -> Float32[np.ndarray, "T El2 Az2"]:
        t, el, az = data.shape
        crop_el = int(el * self.crop_el)
        crop_az = int(az * self.crop_az)

        if crop_el > 0:
            data = data[:, crop_el:-crop_el, :]
        if crop_az > 0:
            data = data[:, :, crop_az:-crop_az]

        if "range_scale" in aug:
            data = (
                data.astype(np.float32) * aug["range_scale"]).astype(np.uint16)
        if aug.get("azimuth_flip"):
            data = np.flip(data, axis=1)

        # Note: the Ouster lidar natively uses mm as a unit, while we use m.
        depth_m = data.astype(np.float32) / 1000
        depth_m[depth_m > self.max_range] = 0.0
        return depth_m / self.max_range
