"""Tests for `nrdk.vis.spectrum`."""

import numpy as np
import torch
from abstract_dataloader.ext.objective import VisualizationConfig

from nrdk.vis import range_azimuth, range_doppler


def _small_spectrum(b=2, d=3, el=2, az=2, rng=4, ch=2, seed=0):
    """Generate a small random spectrum tensor `(b, d, el, az, rng, ch)`."""
    torch.manual_seed(seed)
    return torch.randn(b, d, el, az, rng, ch)


def test_range_doppler_shapes_and_keys():
    """range_doppler returns magnitude, log_magnitude, and phase panels."""
    y_true = _small_spectrum(seed=0)
    y_pred = _small_spectrum(seed=1)
    vis_config = VisualizationConfig(cols=2, width=5, height=4)

    result = range_doppler(y_true, y_pred, vis_config, eps=1e-3)

    assert set(result.keys()) == {"magnitude", "log_magnitude", "phase"}
    for image in result.values():
        assert image.shape == (12, 10, 3)
        assert image.dtype == np.uint8


def test_range_doppler_without_phase():
    """`include_phase=False` omits the phase panel, allowing 1-channel data."""
    y_true = _small_spectrum(ch=1, seed=0)
    y_pred = _small_spectrum(ch=1, seed=1)
    vis_config = VisualizationConfig(cols=2, width=5, height=4)

    result = range_doppler(
        y_true, y_pred, vis_config, eps=1e-3, include_phase=False)

    assert set(result.keys()) == {"magnitude", "log_magnitude"}


def test_range_doppler_eps_avoids_nonfinite_log():
    """`eps` clamping keeps log-magnitude finite even for tiny amplitudes."""
    y_true = _small_spectrum(seed=0) * 1e-6
    y_pred = _small_spectrum(seed=1) * 1e-6
    vis_config = VisualizationConfig(cols=2, width=5, height=4)

    result = range_doppler(y_true, y_pred, vis_config, eps=1e-3)

    for image in result.values():
        assert np.isfinite(image).all()


def test_range_azimuth_shape():
    """range_azimuth returns a single range_azimuth panel."""
    y_true = _small_spectrum(seed=0)
    y_pred = _small_spectrum(seed=1)
    vis_config = VisualizationConfig(cols=2, width=5, height=4)

    result = range_azimuth(y_true, y_pred, vis_config)

    assert set(result.keys()) == {"range_azimuth"}
    assert result["range_azimuth"].shape == (12, 10, 3)
    assert result["range_azimuth"].dtype == np.uint8


def test_range_doppler_default_eps_survives_zero_amplitude():
    """The default `eps` keeps the log panel usable when a bin is exactly 0.

    Without clamping, a single zero amplitude gives `log(0) = -inf`, which
    propagates through `tile_images`' global min/max normalization and
    collapses the whole ground-truth panel to garbage.
    """
    y_true = _small_spectrum(seed=0)
    y_true[0, :, :, :, 0, :] = 0.0
    y_pred = _small_spectrum(seed=1)
    vis_config = VisualizationConfig(cols=2, width=5, height=4)

    default = range_doppler(y_true, y_pred, vis_config)["log_magnitude"]
    unclamped = range_doppler(
        y_true, y_pred, vis_config, eps=0.0)["log_magnitude"]

    # Rows 0-3 are the ground-truth panel; both batch items share a min/max.
    assert np.isfinite(default).all()
    assert np.unique(default[0:4]).size > np.unique(unclamped[0:4]).size
    assert np.unique(default[0:4]).size > 8
