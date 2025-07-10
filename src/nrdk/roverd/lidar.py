"""Lidar transforms."""

from typing import Any, Generic, Mapping, Sequence

import numpy as np
from abstract_dataloader.ext.types import TArray, dataclass
from einops import rearrange
from jaxtyping import Bool, Float, Float64, UInt16
from roverd import types
from roverd.transforms.ouster import ConfigCache, Destagger

from .transforms import SpectrumData


@dataclass
class Occupancy3DData(Generic[TArray]):
    """3D occupancy data.

    Attributes:
        occupancy: 3D occupancy map.
        timestamps: timestamps corresponding to the map data.
    """

    occupancy: Bool[TArray, "batch t el az rng"]
    timestamps: Float64[TArray, "batch t"]


@dataclass
class Occupancy2DData(Generic[TArray]):
    """2D occupancy data.

    Attributes:
        occ: 2D occupancy map.
        timestamps: timestamps corresponding to the map data.
    """

    occupancy: Bool[TArray, "batch t az rng"]
    timestamps: Float64[TArray, "batch t"]


class Occupancy3D:
    """Create 3D occupancy map from Ouster Lidar depth data.

    Augmentations:
        - `azimuth_flip`: flip along azimuth axis.
        - `range_scale`: apply random range scale.

    Args:
        crop_az: fraction of azimuth to crop from both sides of the depth data.
        decimate: decimation factors for elevation, azimuth, and range axes.
        units: lidar range units, in meters; nominal value: `1e-3` (mm).
    """

    def __init__(
        self, crop_az: float = 0.25, decimate: Sequence[int] = (1, 1, 1),
        units: float = 1e-3
    ) -> None:
        self.destagger = Destagger()
        self.crop_az = crop_az
        self.d_el, self.d_az, self.d_rng = decimate
        self.units = units

    def __call__(
        self, lidar: types.OSDepth[np.ndarray],
        radar: SpectrumData[np.ndarray],
        aug: Mapping[str, Any] = {}
    ) -> Occupancy3DData[np.ndarray]:
        """Create 3D occupancy map from Lidar depth data.

        Args:
            lidar: Ouster Lidar depth data with staggered measurements.
            radar: real-valued spectrum data.
            aug: augmentations to apply.

        Returns:
            3D occupancy map.
        """
        rng = self.destagger(lidar).rng

        batch, t, el, az = rng.shape
        crop_az = int(self.crop_az * az)
        rng = rng[:, :, :, crop_az:-crop_az] * self.units

        range_scale = aug.get("range_scale", None)
        if range_scale is not None:
            rng = rng * range_scale
        if aug.get("azimuth_flip", False):
            rng = np.flip(rng, axis=3)

        _batch, _t, _doppler, _el, _az, n_rng, _ch = radar.spectrum.shape
        n_bins = n_rng // self.d_rng
        bin = (rng // (radar.range_resolution * self.d_rng)).astype(np.uint16)
        bin[bin >= n_bins] = 0

        _batch, _t, el2, az2 = bin.shape
        n_el = el2 // self.d_el
        n_az = az2 // self.d_az

        bin = rearrange(
            bin, "batch t (el d_el) (az d_az) -> batch t el az (d_el d_az)",
            el=n_el, d_el=self.d_el, az=n_az, d_az=self.d_az)
        res = np.zeros((batch, t, n_el, n_az, n_bins), dtype=bool)

        i_batch = np.broadcast_to(
            np.arange(batch)[:, None, None, None, None], bin.shape).flatten()
        i_t = np.broadcast_to(
            np.arange(t)[None, :, None, None, None], bin.shape).flatten()
        i_el = np.broadcast_to(
            np.arange(n_el)[None, None, :, None, None], bin.shape).flatten()
        i_az = np.broadcast_to(
            np.arange(n_az)[None, None, None, :, None], bin.shape).flatten()
        res[i_batch, i_t, i_el, i_az, bin.flatten()] = True
        res[:, :, :, :, 0] = False

        return Occupancy3DData(occupancy=res, timestamps=lidar.timestamps)


class Occupancy2D:
    """2D azimuth-range lidar map.

    Follows the procedure used by RadarHD:

    1. Crop to only forward-facing regions; convert to polar coordinates.
    2. Discard points more than 30cm away from the radar plane.
    3. Write points to a 2D polar-coordinate occupancy grid. The range bins
       used in the map match the range bins measured by the radar.

    !!! note

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
        z_min: minimum z-level (in meters) relative to the lidar to include in
            the BEV map.
        z_max: maximum z-level.
            to include in the BEV map, in meters.
        crop_el: crop factor to apply (symmetrically) to each side;
            these values are excluded from BEV calculation. `crop_el` can be
            adjusted to crop nearby objects more to avoid spread due to angled
            objects.
        crop_az: crop factor to apply (symmetrically) to each side; `crop_az`
            should always be `0.25`.
        units: lidar range units, in meters; nominal value: `1e-3` (mm).
    """

    def __init__(
        self, z_min: float = -0.6, z_max: float = 0.9,
        crop_el: float = 0.25, crop_az: float = 0.25, units: float = 1e-3
    ) -> None:
        self.config = ConfigCache()
        self.destagger = Destagger(self.config)

        self.z_min = z_min
        self.z_max = z_max
        self.crop_el = crop_el
        self.crop_az = crop_az
        self.units = units

    def __call__(
        self, lidar: types.OSDepth[np.ndarray],
        radar: SpectrumData[np.ndarray],
        aug: Mapping[str, Any] = {}
    ) -> Occupancy2DData[np.ndarray]:
        """Create 2D occupancy map from Lidar depth data.

        Args:
            lidar: Ouster Lidar depth data with staggered measurements.
            radar: real-valued spectrum data.
            aug: augmentations to apply.

        Returns:
            2D occupancy map.
        """
        rng = self.destagger(lidar).rng

        beam_angles: Float[np.ndarray, "beams"] = np.array(
            self.config[lidar.intrinsics].beam_altitude_angles
        ) / 180 * np.pi

        # Crop, convert mm -> m
        batch, t, el, az = rng.shape
        crop_el = int(self.crop_el * el)
        crop_az = int(self.crop_az * az)
        x_crop = rng[:, :, crop_el:-crop_el, crop_az:-crop_az] * self.units

        if "range_scale" in aug:
            rng = rng * aug["range_scale"]
        if aug.get("azimuth_flip"):
            rng = np.flip(rng, axis=3)

        # Project to polar
        angles = beam_angles[crop_el:-crop_el]
        z = np.sin(angles)[None, None, :, None] * x_crop
        r = np.cos(angles)[None, None, :, None] * x_crop

        # Crop to ex
        r[(z > self.z_max) | (z < self.z_min)] = 0

        # Create map
        _batch, _t, _doppler, _el, _az, n_rng, _ch = radar.spectrum.shape
        bin: UInt16[np.ndarray, "T El Az"] = (
            r // radar.range_resolution).astype(np.uint16)
        bin[bin >= n_rng] = 0

        _batch, _t, _el2, az2 = bin.shape
        res = np.zeros((batch, t, az2, n_rng), dtype=bool)
        i_batch, i_t, i_az = np.meshgrid(
            np.arange(batch), np.arange(t), np.arange(az2), indexing='ij')
        res[
            i_batch[:, :, :, None], i_t[:, :, :, None],
            i_az[:, :, :, None], bin.transpose(0, 1, 3, 2)] = True
        res[:, :, :, 0] = False
        return Occupancy2DData(occupancy=res, timestamps=lidar.timestamps)
