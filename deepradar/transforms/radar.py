"""Radar transforms."""

import numpy as np
from scipy import fft
from skimage.transform import resize

from jaxtyping import Complex64, Int16, Float32
from beartype.typing import Iterable, Any

from .base import BaseTransform


class IIQQtoIQ(BaseTransform):
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


class DiscardTx2(BaseTransform):
    """Discard middle antenna TX2 if data was collected in 3x4 mode."""

    def __call__(
        self, data: Int16[np.ndarray, "D Tx Rx R"], aug: dict[str, Any] = {}
    ) -> Int16[np.ndarray, "D Tx2 Rx R"]:
        if data.shape[1] == 3:
            return data[:, [0, 2]]
        else:
            return data


class AssertTx2(BaseTransform):
    """Assert that the radar data is collected in 3x4 mode."""

    def __call__(
        self, data: Int16[np.ndarray, "D Tx Rx R"], aug: dict[str, Any] = {}
    ) -> Int16[np.ndarray, "D Tx Rx R"]:
        assert data.shape[1] == 3, "Data was not collected in 3x4 mode."
        return data


class BaseFFT(BaseTransform):
    """FFT base class."""

    @classmethod
    def _wrap(
        x: Complex64[np.ndarray, "D A ... R"], width: int
    ) -> Complex64[np.ndarray, "D A ... R"]:
        i_left = x.shape[0] // 2 - width // 2
        i_right = x.shape[0] // 2 + width // 2

        left = x[:i_left]
        center = x[i_left:i_right]
        right = x[i_right:]

        center[:right.shape[0]] += right
        center[-left.shape[0]:] += left

        return center

    def _augment(
        self, data: Complex64[np.ndarray, "D A ... R"],
        aug: dict[str, Any] = {}
    ) -> Complex64[np.ndarray, "D A ... R"]:
        """Apply radar range-doppler data augmentations."""

        if aug.get("azimuth_flip"):
            data = np.flip(data, axis=1)
        if aug.get("doppler_flip"):
            data = np.flip(data, axis=0)
        if aug.get("range_scale", 1.0) != 1.0:
            out_dim = int(aug["range_scale"] * data.shape[-1])
            resized = resize(data, [*data.shape[:-1], out_dim])
            # Upsample -> crop
            if out_dim >= data.shape[-1]:
                data = resized[..., :data.shape[-1]]
            # Downsample -> pad with zeros
            else:
                pad = np.zeros(*data.shape[:-1], data.shape[-1] - out_dim)
                data = np.concatenate([resized, pad])
        if aug.get("speed_scale", 1.0) != 1.0:
            out_dim = 2 * (int(aug["speed_scale"] * data.shape[0]) // 2)
            resized = resize(data, [out_dim, *data.shape[1:]])
            # Downsample -> pad with zeros
            if out_dim <= data.shape[0]:
                pad = np.zeros((data.shape[0] - out_dim) // 2, *data.shape[1:])
                data = np.concatenate((pad, resized, pad), axis=0)
            # Upsample -> wrap around
            else:
                data = BaseFFT._wrap(resized, data.shape[0])

        return data


class FFTLinear(BaseFFT):
    """N-dimensional FFT on a linear array.

    Augmentations:

    - `azimuth_flip`: flip along azimuth axis.
    - `doppler_flip`: flip along doppler axis.
    - `range_scale`: apply random range scale. Excess ranges are cropped;
      missing ranges are zero-filled.
    - `speed_scale`: apply random speed scale. Excess doppler bins are wrapped
      (causing ambiguous doppler velocities); missing doppler velocities are
      zero-filled.

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
    - `range_scale`: apply random range scale. Excess ranges are cropped;
      missing ranges are zero-filled.
    - `speed_scale`: apply random speed scale. Excess doppler bins are wrapped
      (causing ambiguous doppler velocities); missing doppler velocities are
      zero-filled.

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

        return self.augment(daer_shf, aug=aug)


class ComplexParts(BaseTransform):
    """Convert complex numbers to (real, imag) along a new axis.

    Augmentation parameters:

    - `radar_scale`: radar magnitude scale factor.
    """

    def __call__(
        self, data: Complex64[np.ndarray, "..."], aug: dict[str, Any] = {}
    ) -> Float32[np.ndarray, "... 2"]:
        return np.stack(
            [np.real(data), np.imag(data)], axis=-1
        ) / 1e6 * aug.get("radar_scale", 1.0)


class ComplexAmplitude(BaseTransform):
    """Convert complex numbers to amplitude-only.

    Augmentations:

    - `radar_scale`: radar magnitude scale factor.
    """

    def __call__(
        self, data: Complex64[np.ndarray, "..."], aug: dict[str, Any] = {}
    ) -> Float32[np.ndarray, "..."]:
        return np.sqrt(np.abs(data)) / 1e3 * aug.get("radar_scale", 1.0)


class ComplexPhase(BaseTransform):
    """Convert complex numbers to (amplitude, phase) along a new axis.

    Augmentations:

    - `radar_scale`: radar magnitude scale factor.
    - `radar_phase`: radar phase shift.
    """

    def __call__(
        self, data: Complex64[np.ndarray, "..."], aug: dict[str, Any] = {}
    ) -> Float32[np.ndarray, "... 2"]:
        def _normalize(x):
            return (x + np.pi) % (2 * np.pi) - np.pi

        return np.stack([
            np.sqrt(np.abs(data)) / 1e3 * aug.get("radar_scale", 1.0),
            _normalize(np.angle(data) + aug.get("phase_shift", 0.0))
        ], axis=-1)
