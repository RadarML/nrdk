"""Radar FFT-related transforms."""

import numpy as np
from beartype.typing import Any, Iterable
from jaxtyping import Complex64, Int16
from scipy import fft

from .base import Transform


class IIQQtoIQ(Transform):
    """Convert IIQQ i16 raw data to I/Q complex64 data.

    Input shape (MIMO Radar Frames):

    - D: Slow Time
    - Tx: TX antenna index
    - Rx: RX antenna index
    - R2: Fast Time (IIQQ i16 data); is converted to `complex64` (with half
      the length).

    See :py:class:`roverc.radar_api.dca_types.RadarFrame` for details.

    Args:
        scale: scale factor to apply (by multiplication). Since the radar ADC
            reads a (signed) 16-bit `(-32,768, 32,767)`, there is no clear "1"
            value. `1e3` seems about right empirically.
    """

    def __init__(self, path: str, scale: float = 1e-3) -> None:
        self.scale = scale

    def __call__(
        self, data: Int16[np.ndarray, "D Tx Rx R2"], aug: dict[str, Any] = {},
        idx: int = 0
    ) -> Complex64[np.ndarray, "D Tx Rx R"]:
        shape = [*data.shape[:-1], data.shape[-1] // 2]
        iq = np.zeros(shape, dtype=np.complex64)
        iq[..., 0::2] = 1j * data[..., 0::4] + data[..., 2::4]
        iq[..., 1::2] = 1j * data[..., 1::4] + data[..., 3::4]
        return iq * self.scale


class DiscardTx2(Transform):
    """Discard middle antenna TX2 if data was collected in 3x4 mode."""

    def __call__(
        self, data: Int16[np.ndarray, "D Tx Rx R"], aug: dict[str, Any] = {},
        idx: int = 0
    ) -> Int16[np.ndarray, "D Tx2 Rx R"]:
        if data.shape[1] == 3:
            return data[:, [0, 2]]
        else:
            return data


class AssertTx2(Transform):
    """Assert that the radar data is collected in 3x4 mode."""

    def __call__(
        self, data: Int16[np.ndarray, "D Tx Rx R"], aug: dict[str, Any] = {},
        idx: int = 0
    ) -> Int16[np.ndarray, "D Tx Rx R"]:
        assert data.shape[1] == 3, "Data was not collected in 3x4 mode."
        return data


class BaseFFT(Transform):
    """FFT base class."""

    @staticmethod
    def _augment(
        data: Complex64[np.ndarray, "*batch D A ... R"],
        aug: dict[str, Any] = {}, elevation: bool = True
    ) -> Complex64[np.ndarray, "D A ... R"]:
        """Apply radar representation-agnostic data augmentations."""
        if aug.get("azimuth_flip"):
            data = np.flip(data, axis=-3 if elevation else -2)
        if aug.get("doppler_flip"):
            data = np.flip(data, axis=-4 if elevation else -3)
        return data


class FFTPrecomputed(BaseFFT):
    """Pre-computed 4D FFT (only handling augmentations).

    Augmentations:

    - `azimuth_flip`: flip along azimuth axis.
    - `doppler_flip`: flip along doppler axis.
    """

    def __call__(
        self, data: Complex64[np.ndarray, "*batch D A E R"],
        aug: dict[str, Any] = {}, idx: int = 0
    ) -> Complex64[np.ndarray, "*batch D A E R"]:
        return self._augment(data, aug=aug, elevation=True)


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
        aug: dict[str, Any] = {}, idx: int = 0
    ) -> Complex64[np.ndarray, "D A R"]:
        d, tx, rx, r = data.shape
        assert tx == 2, "Only 2-tx mode is supported."

        iq_dar = data.reshape(d, tx * rx, r)

        if self.pad > 0:
            zeros = np.zeros([d, self.pad, r], dtype=np.complex64)
            iq_dar = np.concatenate([iq_dar, zeros], axis=1)

        dar = fft.fftn(iq_dar, axes=self.axes)
        dar_shf = fft.fftshift(dar, axes=[x for x in self.axes if x in (0, 1)])

        return self._augment(dar_shf, aug=aug, elevation=False)


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
        aug: dict[str, Any] = {}, idx: int = 0
    ) -> Complex64[np.ndarray, "D A E R"]:
        d, tx, rx, r = data.shape

        assert tx == 3, "Only 3-tx mode is supported."
        assert rx == 4, "Only 4-rx mode is supported."
        iq_daer = np.zeros((d, 8, 2, r), dtype=np.complex64)
        iq_daer[:, 0:4, 0, :] = data[:, 0, :, :]
        iq_daer[:, 4:8, 0, :] = data[:, 2, :, :]
        iq_daer[:, 2:6, 1, :] = data[:, 1, :, :]

        if self.pad > 0:
            zeros = np.zeros([d, self.pad, 2, r], dtype=np.complex64)
            iq_daer = np.concatenate([iq_daer, zeros], axis=1)

        daer = fft.fftn(iq_daer, axes=self.axes)
        daer_shf = fft.fftshift(
            daer, axes=[x for x in self.axes if x in (0, 1, 2)])

        return self._augment(daer_shf, aug=aug, elevation=True)


class DopplerShuffle(Transform):
    """Shuffle the doppler axis to destroy all doppler information.

    Used only for ablations, obviously don't do this in practice!
    """

    def __call__(
        self, data: Complex64[np.ndarray, "D Tx Rx R"],
        aug: dict[str, Any] = {}, idx: int = 0
    ) -> Complex64[np.ndarray, "D Tx Rx R"]:
        rng = np.random.default_rng()
        rng.shuffle(data, axis=0)
        return data
