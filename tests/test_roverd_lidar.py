"""Tests for lidar transforms over different radar payload types."""

from types import SimpleNamespace

import numpy as np

from nrdk.roverd.lidar import Occupancy3D, _n_range
from roverd import types


def test_n_range_accepts_range_doppler_data() -> None:
    radar = SimpleNamespace(
        rd=np.zeros((1, 1, 64, 3, 4, 256), dtype=np.complex64),
        timestamps=np.zeros((1, 1), dtype=np.float64),
        range_resolution=np.array([0.25], dtype=np.float32),
        doppler_resolution=np.array([0.5], dtype=np.float32),
    )

    assert _n_range(radar) == 256


def test_occupancy3d_accepts_range_doppler_data() -> None:
    lidar = types.OSDepth(
        rng=np.full((1, 1, 4, 8), 4000, dtype=np.uint16),
        timestamps=np.zeros((1, 1), dtype=np.float64),
        intrinsics="",
    )
    radar = SimpleNamespace(
        rd=np.zeros((1, 1, 4, 1, 1, 16), dtype=np.complex64),
        timestamps=np.zeros((1, 1), dtype=np.float64),
        range_resolution=np.array([0.5], dtype=np.float32),
        doppler_resolution=np.array([0.25], dtype=np.float32),
    )

    transform = Occupancy3D(crop_az=0.25, decimate=(2, 2, 4), units=1e-3)
    transform.destagger = lambda data: data

    out = transform(lidar=lidar, radar=radar, aug={})

    assert out.occupancy.shape == (1, 1, 2, 2, 4)
    assert out.timestamps.shape == (1, 1)
