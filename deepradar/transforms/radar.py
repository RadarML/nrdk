"""Radar transforms."""

import json
import os

import numpy as np
import torch
from beartype.typing import Any, Iterable
from jaxtyping import Complex64, Float32, Int16
from scipy import fft
from torchvision import transforms

from .base import Transform


class RadarResolution(Transform):
    """Get radar resolution metadata.

    Augmentations:

    - `range_scale`: apply scale multiplicatively to `range_resolution`.
    - `speed_scale`: apply multiplicatively to `doppler_resolution`.

    Args:
        path: path to dataset directory.
    """

    def __init__(self, path: str) -> None:

        with open(os.path.join(path, "radar", "radar.json")) as f:
            cfg = json.load(f)
        self.range_res = cfg["range_resolution"]
        self.doppler_res = cfg["doppler_resolution"]

    def __call__(
        self, _, aug: dict[str, Any] = {}
    ) -> Float32[np.ndarray, "d2"]:

        meta = np.array([
            self.range_res * aug.get("range_scale", 1.0),
            self.doppler_res * aug.get("speed_scale", 1.0)
        ], dtype=np.float32)
        return meta


class IIQQtoIQ(Transform):
    """Convert IIQQ i16 raw data to I/Q complex64 data.

    Input shape (MIMO Radar Frames):

    - D: Slow Time
    - Tx: TX antenna index
    - Rx: RX antenna index
    - R2: Fast Time (IIQQ i16 data); is converted to `complex64` (with half
      the length).

    See `collect.radar_api.dca_types.RadarFrame` for details.
    """

    def __call__(
        self, data: Int16[np.ndarray, "D Tx Rx R2"], aug: dict[str, Any] = {}
    ) -> Complex64[np.ndarray, "D Tx Rx R"]:
        shape = [*data.shape[:-1], data.shape[-1] // 2]
        iq = np.zeros(shape, dtype=np.complex64)
        iq[..., 0::2] = 1j * data[..., 0::4] + data[..., 2::4]
        iq[..., 1::2] = 1j * data[..., 1::4] + data[..., 3::4]
        return iq


class DiscardTx2(Transform):
    """Discard middle antenna TX2 if data was collected in 3x4 mode."""

    def __call__(
        self, data: Int16[np.ndarray, "D Tx Rx R"], aug: dict[str, Any] = {}
    ) -> Int16[np.ndarray, "D Tx2 Rx R"]:
        if data.shape[1] == 3:
            return data[:, [0, 2]]
        else:
            return data


class AssertTx2(Transform):
    """Assert that the radar data is collected in 3x4 mode."""

    def __call__(
        self, data: Int16[np.ndarray, "D Tx Rx R"], aug: dict[str, Any] = {}
    ) -> Int16[np.ndarray, "D Tx Rx R"]:
        assert data.shape[1] == 3, "Data was not collected in 3x4 mode."
        return data


class BaseFFT(Transform):
    """FFT base class."""

    @staticmethod
    def _augment(
        data: Complex64[np.ndarray, "D A ... R"], aug: dict[str, Any] = {}
    ) -> Complex64[np.ndarray, "D A ... R"]:
        """Apply radar representation-agnostic data augmentations."""
        if aug.get("azimuth_flip"):
            data = np.flip(data, axis=1)
        if aug.get("doppler_flip"):
            data = np.flip(data, axis=0)
        return data


class FFTLinear(BaseFFT):
    """N-dimensional FFT on a linear array.

    Augmentations:

    - `azimuth_flip`: flip along azimuth axis.
    - `doppler_flip`: flip along doppler axis.

    Args:
        pad: azimuth padding; output has shape (tx * rx + pad).
        axes: axes to apply an FFT to; 0=doppler, 1=azimuth, 2=range.
    """

    def __init__(
        self, path: str, pad: int = 0, axes: Iterable[int] = (0, 1, 2)
    ) -> None:
        self.pad = pad
        self.axes = axes

    def __call__(
        self, data: Complex64[np.ndarray, "D Tx Rx R"],
        aug: dict[str, Any] = {}
    ) -> Complex64[np.ndarray, "D A R"]:
        d, tx, rx, r = data.shape
        assert tx == 2, "Only 2-tx mode is supported."

        iq_dar = data.reshape(d, tx * rx, r)

        if self.pad > 0:
            zeros = np.zeros([d, self.pad, r], dtype=np.complex64)
            iq_dar = np.concatenate([iq_dar, zeros], axis=1)

        dar = fft.fftn(iq_dar, axes=self.axes)
        dar_shf = fft.fftshift(dar, axes=[x for x in self.axes if x in (0, 1)])

        return self._augment(dar_shf, aug=aug)


class FFTArray(BaseFFT):
    """N-dimensional FFT on a nonlinear array.

    Augmentations:

    - `azimuth_flip`: flip along azimuth axis.
    - `doppler_flip`: flip along doppler axis.

    Args:
        pad: azimuth padding; output has shape (tx * rx + pad).
        axes: axes to apply an FFT to; 0=doppler, 1=azimuth, 2=elevation,
            3=range.
    """

    def __init__(
        self, path: str, pad: int = 0, axes: Iterable[int] = (0, 1, 2, 3)
    ) -> None:
        self.pad = pad
        self.axes = axes

    def __call__(
        self, data: Complex64[np.ndarray, "D Tx Rx R"],
        aug: dict[str, Any] = {}
    ) -> Complex64[np.ndarray, "D A E R"]:
        d, tx, rx, r = data.shape

        assert tx == 3, "Only 3-tx mode is supported."
        assert rx == 4, "Only 4-rx mode is supported."
        iq_daer = np.zeros((d, 8, 2, r), dtype=np.complex64)
        iq_daer[:, :4, 0, :] = data[:, 0, :, :]
        iq_daer[:, 4:8, 0, :] = data[:, 2, :, :]
        iq_daer[:, 2:6, 1, :] = data[:, 1, :, :]

        if self.pad > 0:
            zeros = np.zeros([d, self.pad, 2, r], dtype=np.complex64)
            iq_daer = np.concatenate([iq_daer, zeros], axis=1)

        daer = fft.fftn(iq_daer, axes=self.axes)
        daer_shf = fft.fftshift(
            daer, axes=[x for x in self.axes if x in (0, 1, 2)])

        return self._augment(daer_shf, aug=aug)


class DopplerShuffle(Transform):
    """Shuffle the doppler axis to destroy all doppler information.

    Used only for ablations, obviously don't do this in practice!
    """

    def __call__(
        self, data: Complex64[np.ndarray, "D Tx Rx R"],
        aug: dict[str, Any] = {}
    ) -> None:
        rng = np.random.default_rng()
        return rng.shuffle(data, axis=0)


class Representation(Transform):
    """Base class for radar representations."""

    @staticmethod
    def _wrap(
        x: Float32[np.ndarray, "D A ... R"], width: int
    ) -> Float32[np.ndarray, "D_crop A ... R"]:
        i_left = x.shape[0] // 2 - width // 2
        i_right = x.shape[0] // 2 + width // 2

        left = x[:i_left]
        center = x[i_left:i_right]
        right = x[i_right:]

        center[:right.shape[0]] += right
        center[-left.shape[0]:] += left

        return center

    @classmethod
    def _augment(
        cls, data: Float32[np.ndarray, "D A ... R"],
        aug: dict[str, Any] = {}
    ) -> Float32[np.ndarray, "D A ... R"]:
        """Apply radar representation-specific data augmentations.

        NOTE: we use `torchvision.transforms.Resize`, which requires a
        round-trip through a (cpu) `Tensor`. From some limited testing, this
        appears to be the most performant image resizing which supports
        antialiasing, with `skimage.transform.resize` being particularly slow.
        """
        Nd = data.shape[0]
        Nr = data.shape[-1]
        range_out_dim = int(aug.get("range_scale", 1.0) * Nr)
        speed_out_dim = 2 * (int(aug.get("speed_scale", 1.0) * Nd) // 2)

        if range_out_dim != Nr or speed_out_dim != Nd:
            resized = transforms.Resize(
                (range_out_dim, speed_out_dim),
                interpolation=transforms.InterpolationMode.BILINEAR,
                antialias=True
            )(torch.Tensor(np.moveaxis(data, 0, -1)))
            resized = np.moveaxis(resized.numpy(), -1, 0)

            # Upsample -> crop
            if range_out_dim >= Nr:
                resized = resized[..., :Nr]
            # Downsample -> zero pad far ranges (high indices)
            else:
                pad = np.zeros(
                    (*data.shape[:-1], Nr - range_out_dim), dtype=np.float32)
                resized = np.concatenate([resized, pad], axis=-1)

            # Upsample -> wrap
            if speed_out_dim > data.shape[0]:
                resized = cls._wrap(resized, Nd)
            # Downsample -> zero pad high velocities (low and high indices)
            else:
                pad = np.zeros(
                    ((Nd - speed_out_dim) // 2, *data.shape[1:]),
                    dtype=np.float32)
                resized = np.concatenate((pad, resized, pad), axis=0)

            data = resized

        return data


class ComplexParts(Representation):
    """Convert complex numbers to (real, imag) along a new axis.

    Augmentations:

    - `radar_scale`: radar magnitude scale factor.
    - `radar_phase`: radar phase shift.
    - `range_scale`: apply random range scale. Excess ranges are cropped;
      missing ranges are zero-filled.
    - `speed_scale`: apply random speed scale. Excess doppler bins are wrapped
      (causing ambiguous doppler velocities); missing doppler velocities are
      zero-filled.
    """

    def __call__(
        self, data: Complex64[np.ndarray, "..."], aug: dict[str, Any] = {}
    ) -> Float32[np.ndarray, "... 2"]:
        if aug.get("radar_phase"):
            data *= np.exp(-1j * aug["radar_phase"])

        stretched = [
            self._augment(np.real(data), aug),
            self._augment(np.imag(data), aug)]
        return np.stack(stretched, axis=-1) / 1e6 * aug.get("radar_scale", 1.0)


class ComplexAmplitude(Representation):
    """Convert complex numbers to amplitude-only.

    Augmentations:

    - `radar_scale`: radar magnitude scale factor.
    - `range_scale`: apply random range scale. Excess ranges are cropped;
      missing ranges are zero-filled.
    - `speed_scale`: apply random speed scale. Excess doppler bins are wrapped
      (causing ambiguous doppler velocities); missing doppler velocities are
      zero-filled.
    """

    def __call__(
        self, data: Complex64[np.ndarray, "..."], aug: dict[str, Any] = {}
    ) -> Float32[np.ndarray, "..."]:
        stretched = self._augment(np.sqrt(np.abs(data)), aug)
        return stretched / 1e3 * aug.get("radar_scale", 1.0)


class ComplexPhase(Representation):
    """Convert complex numbers to (amplitude, phase) along a new axis.

    Augmentations:

    - `radar_scale`: radar magnitude scale factor.
    - `radar_phase`: radar phase shift.
    - `range_scale`: apply random range scale. Excess ranges are cropped;
      missing ranges are zero-filled.
    - `speed_scale`: apply random speed scale. Excess doppler bins are wrapped
      (causing ambiguous doppler velocities); missing doppler velocities are
      zero-filled.
    """

    def __call__(
        self, data: Complex64[np.ndarray, "..."], aug: dict[str, Any] = {}
    ) -> Float32[np.ndarray, "... 2"]:
        def _normalize(x):
            return (x + np.pi) % (2 * np.pi) - np.pi

        stretched_magnitude = self._augment(np.sqrt(np.abs(data)), aug)
        stretched_phase = self._augment(np.angle(data), aug)

        return np.stack([
            stretched_magnitude / 1e3 * aug.get("radar_scale", 1.0),
            _normalize(stretched_phase + aug.get("phase_shift", 0.0))
        ], axis=-1)
