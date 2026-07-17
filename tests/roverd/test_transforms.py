"""Tests for `nrdk.roverd.transforms`."""

import numpy as np
import pytest
from roverd import types
from xwr.nn import Magnitude
from xwr.rsp.numpy import AWR1843Boost

from nrdk.roverd.transforms import Semseg, Spectrum, Velocity


def _small_iq(b=1, t=2, slow=4, tx=3, rx=4, fast=16, seed=0):
    """Generate small int16 IIQQ-interleaved I/Q data.

    AWR1843Boost's virtual array requires exactly `tx=3, rx=4`.
    """
    rng = np.random.default_rng(seed)
    iq = rng.integers(
        -100, 100, size=(b, t, slow, tx, rx, fast), dtype=np.int16)
    return types.XWRRadarIQ(
        iq=iq,
        timestamps=np.arange(b * t, dtype=np.float64).reshape(b, t),
        range_resolution=np.full((b,), 0.1),
        doppler_resolution=np.full((b,), 0.05),
        valid=np.ones((b, t), dtype=np.uint8))


def _pose(vel, rot=None, b=1, t=1):
    vel = np.asarray(vel, dtype=np.float32).reshape(b, t, 3)
    if rot is None:
        rot = np.broadcast_to(np.eye(3, dtype=np.float32), (b, t, 3, 3))
    return types.Pose(
        pos=np.zeros((b, t, 3), dtype=np.float32),
        vel=vel,
        acc=np.zeros((b, t, 3), dtype=np.float32),
        rot=np.array(rot, dtype=np.float32).reshape(b, t, 3, 3),
        timestamps=np.arange(b * t, dtype=np.float64).reshape(b, t))


def _radar(doppler_resolution=0.1, b=1):
    from nrdk.roverd.transforms import SpectrumData
    return SpectrumData(
        spectrum=np.zeros((b, 1, 1, 1, 1, 1, 1), dtype=np.float32),
        timestamps=np.zeros((b, 1)),
        range_resolution=np.full((b,), 1.0),
        doppler_resolution=np.full((b,), doppler_resolution))


def test_spectrum():
    """Check metadata passthrough and spectrum correctness.

    The spectrum should match manually flattening/unflattening the
    `batch, t` axes around `rsp` + `rep`.
    """
    iq = _small_iq()
    rsp = AWR1843Boost()
    rep = Magnitude()
    transform = Spectrum(rsp=rsp, rep=rep)

    out = transform(iq)

    b, t = iq.iq.shape[:2]
    flat_iq = iq.iq.reshape(-1, *iq.iq.shape[2:])
    direct = rep(rsp(flat_iq))
    direct = direct.reshape(b, t, *direct.shape[1:])

    assert out.spectrum.shape[-1] == 1  # Magnitude -> single channel
    assert np.all(np.isfinite(out.spectrum))
    np.testing.assert_allclose(out.spectrum, direct)
    np.testing.assert_array_equal(out.timestamps, iq.timestamps)
    np.testing.assert_array_equal(out.range_resolution, iq.range_resolution)
    np.testing.assert_array_equal(
        out.doppler_resolution, iq.doppler_resolution)


def test_semseg_azimuth_flip():
    """Check passthrough and the `azimuth_flip` augmentation.

    Without augmentation, the array passes through unchanged (same object).
    With `azimuth_flip`, the width (last) axis is mirrored and copied
    (flipped views have negative strides).
    """
    semseg = types.CameraSemseg(
        semseg=np.arange(2 * 4, dtype=np.uint8).reshape(1, 1, 2, 4),
        timestamps=np.array([[0.0]]))
    transform = Semseg()

    out = transform(semseg)
    assert out.semseg is semseg.semseg
    np.testing.assert_array_equal(out.timestamps, semseg.timestamps)

    flipped = transform(semseg, aug={"azimuth_flip": True})
    np.testing.assert_array_equal(
        flipped.semseg, np.flip(semseg.semseg, axis=3))
    assert flipped.semseg.base is None


def test_velocity_rotation_applies_inverse():
    """Check that `rot` is undone via `inv(rot)`.

    A 90-degree rotation about z applied to world-frame `vel=[1, 0, 0]`
    should yield sensor-frame `[0, -1, 0]`.
    """
    theta = np.pi / 2
    rot = [
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1],
    ]
    pose = _pose(vel=[1.0, 0.0, 0.0], rot=rot)

    out = Velocity()(pose)

    expected = np.linalg.inv(np.array(rot)) @ np.array([1.0, 0.0, 0.0])
    np.testing.assert_allclose(out.vel[0, 0], expected, atol=1e-6)
    np.testing.assert_array_equal(out.timestamps, pose.timestamps)


def test_velocity_flips():
    """Check the `doppler_flip` and `azimuth_flip` augmentations.

    `doppler_flip` negates the whole vector. `azimuth_flip` negates only
    index 1 (y), per the `Velocity` docstring.
    """
    pose = _pose(vel=[0.5, -1.0, 2.0])
    transform = Velocity()

    doppler_out = transform(pose, aug={"doppler_flip": True})
    azimuth_out = transform(pose, aug={"azimuth_flip": True})

    np.testing.assert_allclose(doppler_out.vel, -pose.vel)
    np.testing.assert_allclose(azimuth_out.vel[0, 0], [0.5, 1.0, 2.0])


@pytest.mark.parametrize("aug,doppler_resolution,scale", [
    ({}, None, 1.0),
    ({"speed_scale": 2.0}, None, 2.0),
    ({}, 0.5, 1 / 0.5),
    ({"speed_scale": 0.25}, 0.2, 0.25 / 0.2),
])
def test_velocity_speed_scale_and_radar(aug, doppler_resolution, scale):
    """Check that `speed_scale` and radar `doppler_resolution` scale `vel`.

    Both scale linearly, and compose when both are given.
    """
    pose = _pose(vel=[2.0, 0.0, -1.0])
    radar = (
        _radar(doppler_resolution=doppler_resolution)
        if doppler_resolution is not None else None)

    out = Velocity()(pose, radar=radar, aug=aug)

    np.testing.assert_allclose(out.vel, pose.vel * scale)
