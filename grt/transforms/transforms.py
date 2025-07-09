"""Data transforms."""

from typing import Any, Generic, Mapping

import numpy as np
from abstract_dataloader.ext.types import TArray, dataclass
from abstract_dataloader.spec import Transform
from jaxtyping import Complex64, Float, Float32, Float64
from roverd import types

from xwr import nn as xwr_nn
from xwr import rsp as xwr_rsp


@dataclass
class RealSpectrum(Generic[TArray]):
    """Real 4D spectrum.

    Attributes:
        spectrum: real-valued 4D spectrum with a leading batch axis and
            trailing channel axis.
        timestamps: timestamps corresponding to the spectrum.
        range_resolution: range resolution of the spectrum.
        doppler_resolution: doppler resolution of the spectrum.
    """

    spectrum: Float32[TArray, "batch t doppler el az rng ch"]
    timestamps: Float64[TArray, "batch t"]
    range_resolution: Float[TArray, "batch"]
    doppler_resolution: Float[TArray, "batch"]


@dataclass
class Velocity(Generic[TArray]):
    """Ego-velocity with a front-left-up (FLU) coordinate convention.

    Attributes:
        vel: relative velocity in the sensor's frame of reference.
        timestamps: timestamps corresponding to the velocity.
    """

    vel: Float[TArray, "batch t 3"]
    timestamps: Float[TArray, "batch t"]


class Spectrum(Transform[
    types.XWRRadarIQ[np.ndarray], RealSpectrum[np.ndarray]
]):
    """Transform raw I/Q data to 4D spectrum data via FFT.

    Augmentations:
        - `azimuth_flip`: flip along azimuth axis.
        - `doppler_flip`: flip along doppler axis.
        - `radar_scale`: radar magnitude scale factor.
        - `radar_phase`: radar phase shift.
        - `range_scale`: apply random range scale. Excess ranges are cropped;
          missing ranges are zero-filled.
        - `speed_scale`: apply random speed scale. Excess doppler bins are
          wrapped (causing ambiguous doppler velocities); missing doppler
          velocities are zero-filled.

    Args:
        rsp: Radar signal processing callable to apply.
        rep: real-value representation to apply to the complex spectrum.
    """

    def __init__(
        self, rsp: xwr_rsp.RSP[np.ndarray], rep: xwr_nn.Representation
    ) -> None:
        self.rsp = rsp
        self.rep = rep

    def __call__(
        self, iq: types.XWRRadarIQ[np.ndarray], aug: Mapping[str, Any] = {}
    ) -> RealSpectrum[np.ndarray]:
        """Process radar data.

        Args:
            iq: input data.
            aug: augmentations to apply.

        Returns:
            Computed real spectrum, with a varying number of trailing axis
                channels depending on the provided `rep`.
        """
        cplx: Complex64[np.ndarray, "batch t doppler el az rng"]
        _iq = iq.iq.reshape(-1, *iq.iq.shape[2:])
        cplx = self.rsp(_iq)

        real: Float32[np.ndarray, "batch t doppler el az rng ch"]
        _real = self.rep(cplx, aug)
        real = _real.reshape(*iq.iq.shape[:2], *_real.shape[1:])

        return RealSpectrum(
            spectrum=real,
            timestamps=iq.timestamps,
            range_resolution=iq.range_resolution,
            doppler_resolution=iq.doppler_resolution)


class Semseg(Transform[
    types.CameraSemseg[np.ndarray], types.CameraSemseg[np.ndarray]
]):
    """Handle camera-related augmentations.

    Augmentations:
        - `azimuth_flip`: flip the camera left/right.
    """

    def __call__(
        self, semseg: types.CameraSemseg[np.ndarray],
        aug: Mapping[str, Any] = {}
    ) -> types.CameraSemseg[np.ndarray]:
        """Apply camera-related augmentations.

        Args:
            semseg: input semantic segmentation data.
            aug: augmentations to apply.

        Returns:
            Semseg image with azimuth flipped if specified.
        """
        if aug.get("azimuth_flip", False):
            # A copy is required here since torch doesn't allow creating
            # tensors from data with negative stride.
            # semseg: batch t height width
            img = np.flip(semseg.semseg, axis=3).copy()
        else:
            img = semseg.semseg

        return types.CameraSemseg(semseg=img, timestamps=semseg.timestamps)


class RelativeVelocity(Transform[
    types.Pose[np.ndarray], Velocity[np.ndarray]
]):
    """Velocity relative to the sensor's frame of reference.

    !!! note "Implementation Notes"

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

    def __call__(
        self, pose: types.Pose[np.ndarray],
        radar: RealSpectrum[np.ndarray] | None = None,
        aug: Mapping[str, Any] = {}
    ) -> Velocity[np.ndarray]:
        """Compute relative velocity.

        Args:
            pose: input pose.
            radar: radar spectrum data, used to scale the velocity by the
                doppler resolution if provided.
            aug: data augmentations.

        Returns:
            Ego-velocity in the sensor's frame of reference.
        """
        scale = aug.get("speed_scale", 1.0)
        vel: Float[np.ndarray, "batch t 3"] = np.matmul(
            np.linalg.inv(pose.rot), pose.vel[..., None]
        ).squeeze(-1) * scale

        if aug.get("doppler_flip", False):
            vel *= -1
        if aug.get("azimuth_flip", False):
            vel[..., 2] = vel[..., 2] * -1

        if radar is not None:
            vel = vel / radar.doppler_resolution[:, None, None]

        return Velocity(vel=vel, timestamps=pose.timestamps)
