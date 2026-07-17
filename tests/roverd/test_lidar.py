"""Tests for `nrdk.roverd.lidar`."""

import json

import numpy as np
from roverd import types

from nrdk.roverd.lidar import Occupancy2D, Occupancy3D
from nrdk.roverd.transforms import SpectrumData


def _write_intrinsics(
    tmp_path, beam_altitude_angles, n_beams=8, n_ticks=8
) -> str:
    """Write a minimal `lidar.json`-style intrinsics file.

    Uses a zero `pixel_shift_by_row`, which makes `Destagger` a no-op, so
    tests can reason directly about the raw `rng` array without also having
    to account for the destaggering shift.
    """
    intrinsics = {
        "lidar_data_format": {
            "pixels_per_column": n_beams,
            "columns_per_frame": n_ticks,
            "pixel_shift_by_row": [0] * n_beams,
        },
        "beam_intrinsics": {
            "beam_altitude_angles": list(beam_altitude_angles),
            "beam_azimuth_angles": [0.0] * n_beams,
            "beam_to_lidar_transform": list(np.eye(4).flatten()),
        },
        "lidar_intrinsics": {
            "lidar_to_sensor_transform": list(np.eye(4).flatten()),
        },
    }
    path = tmp_path / "lidar.json"
    path.write_text(json.dumps(intrinsics))
    return str(path)


def _make_lidar(rng: np.ndarray, intrinsics: str) -> types.OSDepth:
    batch, t, _el, _az = rng.shape
    return types.OSDepth(
        rng=rng.astype(np.uint16),
        timestamps=np.arange(batch * t, dtype=np.float64).reshape(batch, t),
        intrinsics=intrinsics)


def _make_spectrum(
    n_rng: int, range_resolution: float = 1.0,
    doppler_resolution: float = 0.05
) -> SpectrumData:
    return SpectrumData(
        spectrum=np.zeros((1, 1, 1, 1, 1, n_rng, 1), dtype=np.float32),
        timestamps=np.array([[0.0]]),
        range_resolution=np.array([range_resolution]),
        doppler_resolution=np.array([doppler_resolution]))


def test_occupancy3d_bin_positions_and_decimation(tmp_path):
    """Bin indices and el/az decimation grouping match a hand-derived map.

    Beams 8, ticks 8, `crop_az=0.25` crops 2 columns off each side (keeping
    raw azimuth indices 2-5), `decimate=(2, 2, 1)` groups consecutive pairs
    of beams and (cropped) azimuth columns together. `units=1.0` and
    `range_resolution=1.0` make `rng` values map directly onto bin indices.

    Three points are placed:

    - beam 0, raw az 2 (-> el group 0, az group 0): range 3 -> bin 3.
    - beam 1, raw az 3 (-> el group 0, az group 0): range 5 -> bin 5.
    - beam 2, raw az 4 (-> el group 1, az group 1): range 6 -> bin 6.

    Beams 0-1 fall into the same decimated el group, so the first two points
    land in the same output cell (demonstrating multiple detections
    surviving decimation independently); everything else is zero and must
    map to bin 0, which is always cleared.
    """
    intrinsics = _write_intrinsics(tmp_path, beam_altitude_angles=[0] * 8)
    rng = np.zeros((1, 1, 8, 8), dtype=np.uint16)
    rng[0, 0, 0, 2] = 3
    rng[0, 0, 1, 3] = 5
    rng[0, 0, 2, 4] = 6
    lidar = _make_lidar(rng, intrinsics)
    radar = _make_spectrum(n_rng=8, range_resolution=1.0)

    transform = Occupancy3D(crop_az=0.25, decimate=(2, 2, 1), units=1.0)
    out = transform(lidar, radar, aug={})

    assert out.occupancy.shape == (1, 1, 4, 2, 8)
    np.testing.assert_array_equal(out.timestamps, lidar.timestamps)

    expected = np.zeros((4, 2, 8), dtype=bool)
    expected[0, 0, 3] = True
    expected[0, 0, 5] = True
    expected[1, 1, 6] = True
    np.testing.assert_array_equal(out.occupancy[0, 0], expected)


def test_occupancy3d_range_scale_augmentation(tmp_path):
    """`range_scale` multiplies range before binning; overflow maps to 0."""
    intrinsics = _write_intrinsics(tmp_path, beam_altitude_angles=[0] * 8)
    rng = np.zeros((1, 1, 8, 8), dtype=np.uint16)
    rng[0, 0, 0, 2] = 3
    rng[0, 0, 1, 3] = 5
    rng[0, 0, 2, 4] = 6
    lidar = _make_lidar(rng, intrinsics)
    radar = _make_spectrum(n_rng=8, range_resolution=1.0)

    transform = Occupancy3D(crop_az=0.25, decimate=(2, 2, 1), units=1.0)
    out = transform(lidar, radar, aug={"range_scale": 2.0})

    expected = np.zeros((4, 2, 8), dtype=bool)
    expected[0, 0, 6] = True
    np.testing.assert_array_equal(out.occupancy[0, 0], expected)


def test_occupancy3d_azimuth_flip_augmentation(tmp_path):
    """`azimuth_flip` mirrors the (cropped) azimuth axis before binning."""
    intrinsics = _write_intrinsics(tmp_path, beam_altitude_angles=[0] * 8)
    rng = np.zeros((1, 1, 8, 8), dtype=np.uint16)
    rng[0, 0, 0, 2] = 3
    rng[0, 0, 1, 3] = 5
    rng[0, 0, 2, 4] = 6
    lidar = _make_lidar(rng, intrinsics)
    radar = _make_spectrum(n_rng=8, range_resolution=1.0)

    transform = Occupancy3D(crop_az=0.25, decimate=(2, 2, 1), units=1.0)
    out = transform(lidar, radar, aug={"azimuth_flip": True})

    expected = np.zeros((4, 2, 8), dtype=bool)
    expected[0, 1, 3] = True
    expected[0, 1, 5] = True
    expected[1, 0, 6] = True
    np.testing.assert_array_equal(out.occupancy[0, 0], expected)


def test_occupancy2d_projection_clipping_and_overflow(tmp_path):
    """Polar projection, z-level clipping, and range overflow all apply.

    Beam altitudes (only beams 2-5 survive `crop_el=0.25` on 8 beams) are
    `[0, 30, -80, 0]` degrees; `units=0.1` turns integer `rng` values into
    fractional meters. With `z_min=-0.5, z_max=0.5`:

    - beam 2 (angle 0), az col 0: `rng=4 -> x=0.4`, `z=0` (in bounds),
      `r=0.4 -> bin=int(0.4/0.05)=8`.
    - beam 3 (angle 30), az col 1: `rng=8 -> x=0.8`, `z=0.5*0.8=0.4` (in
      bounds), `r=cos(30)*0.8~=0.693 -> bin=13`.
    - beam 4 (angle -80), same az col 1: `rng=10 -> x=1.0`,
      `z=sin(-80)*1.0~=-0.985` (out of `[-0.5, 0.5]`) -> clipped to `r=0`,
      contributing nothing (elevation is a pure column-wise OR, so col 1
      only shows beam 3's bin 13).
    - beam 5 (angle 0), az col 2: `rng=500 -> x=50 -> r=50`, whose bin
      (1000) exceeds `n_rng=20` and is zeroed.
    - az col 3 has no points at all.
    """
    intrinsics = _write_intrinsics(
        tmp_path, beam_altitude_angles=[0, 0, 0, 30, -80, 0, 0, 0])
    rng = np.zeros((1, 1, 8, 8), dtype=np.uint16)
    rng[0, 0, 2, 2] = 4
    rng[0, 0, 3, 3] = 8
    rng[0, 0, 4, 3] = 10
    rng[0, 0, 5, 4] = 500
    lidar = _make_lidar(rng, intrinsics)
    radar = _make_spectrum(n_rng=20, range_resolution=0.05)

    transform = Occupancy2D(
        z_min=-0.5, z_max=0.5, crop_el=0.25, crop_az=0.25, units=0.1)
    out = transform(lidar, radar, aug={})

    assert out.occupancy.shape == (1, 1, 4, 20)
    np.testing.assert_array_equal(out.timestamps, lidar.timestamps)

    expected = np.zeros((4, 20), dtype=bool)
    expected[0, 8] = True
    expected[1, 13] = True
    np.testing.assert_array_equal(out.occupancy[0, 0], expected)


def test_occupancy2d_range_scale_augmentation(tmp_path):
    """`range_scale` multiplies range before binning; overflow maps to 0.

    All beam angles are 0, so `z=0` (never clipped) and `r == x`. Doubling
    ranges moves point 1 (range 4 -> x=0.8, bin 16) and overflows point 2
    (range 6 -> x=1.2, bin 24 >= `n_rng=20`), which is zeroed out.
    """
    intrinsics = _write_intrinsics(tmp_path, beam_altitude_angles=[0] * 8)
    rng = np.zeros((1, 1, 8, 8), dtype=np.uint16)
    rng[0, 0, 2, 2] = 4
    rng[0, 0, 3, 4] = 6
    lidar = _make_lidar(rng, intrinsics)
    radar = _make_spectrum(n_rng=20, range_resolution=0.05)

    transform = Occupancy2D(
        z_min=-0.5, z_max=0.5, crop_el=0.25, crop_az=0.25, units=0.1)
    out = transform(lidar, radar, aug={"range_scale": 2.0})

    expected = np.zeros((4, 20), dtype=bool)
    expected[0, 16] = True
    np.testing.assert_array_equal(out.occupancy[0, 0], expected)


def test_occupancy2d_azimuth_flip_augmentation(tmp_path):
    """`azimuth_flip` mirrors the (cropped) azimuth axis before binning."""
    intrinsics = _write_intrinsics(tmp_path, beam_altitude_angles=[0] * 8)
    rng = np.zeros((1, 1, 8, 8), dtype=np.uint16)
    rng[0, 0, 2, 2] = 4
    rng[0, 0, 3, 4] = 6
    lidar = _make_lidar(rng, intrinsics)
    radar = _make_spectrum(n_rng=20, range_resolution=0.05)

    transform = Occupancy2D(
        z_min=-0.5, z_max=0.5, crop_el=0.25, crop_az=0.25, units=0.1)
    out = transform(lidar, radar, aug={"azimuth_flip": True})

    expected = np.zeros((4, 20), dtype=bool)
    expected[3, 8] = True
    expected[1, 12] = True
    np.testing.assert_array_equal(out.occupancy[0, 0], expected)
