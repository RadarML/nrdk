"""Radar FFT-related transforms."""

import numpy as np
from beartype.typing import Any, Iterable
from jaxtyping import Complex64, Int16
from scipy import fft

from .base import Transform


class IIQQtoIQ(Transform):
    """Convert TI IIQQ-i16 raw data to I/Q complex64 data.

    While the nominal data has a 4D structure (see :py:class:`.FFTArray`),
    this transform only assumes that the data has a trailing IIQQ-i16 axis,
    which is converted to a complex64 axis of half the numerical length
    (due to combining real and complex parts into a single data type).

    See :py:class:`roverc.radar_api.dca_types.RadarFrame` for details about
    the IIQQ-i16 data type.

    Args:
        scale: scale factor to apply (by multiplication). Since the radar ADC
            reads a (signed) 16-bit `(-32,768, 32,767)`, there is no clear "1"
            value. `1e3` seems about right empirically.
    """

    def __init__(self, path: str, scale: float = 1e-3) -> None:
        self.scale = scale

    def __call__(
        self, data: Int16[np.ndarray, "... R2"], aug: dict[str, Any] = {},
        idx: int = 0
    ) -> Complex64[np.ndarray, "... R"]:
        """Apply transform.

        Args:
            data: input data. Must have a trailing IIQQ-i16 axis.

        Returns:
            Output with the same shape as `data`, except with the trailing axis
            collapsed to a `complex64` type.
        """
        shape = [*data.shape[:-1], data.shape[-1] // 2]
        iq = np.zeros(shape, dtype=np.complex64)
        iq[..., 0::2] = 1j * data[..., 0::4] + data[..., 2::4]
        iq[..., 1::2] = 1j * data[..., 1::4] + data[..., 3::4]
        return iq * self.scale


class DiscardTx2(Transform):
    """Discard middle antenna TX2 if data was collected in 3x4 mode."""

    def __call__(
        self, data: Int16[np.ndarray, "T D Tx Rx R"], aug: dict[str, Any] = {},
        idx: int = 0
    ) -> Int16[np.ndarray, "T D Tx2 Rx R"]:
        """Apply transform.

        Args:
            data: input 4D data cubes in batch-slow-tx-rx-fast order.

        Returns:
            If the data was collected in 3x4 mode, the middle antenna is
            discarded. Otherwise, this transform is the identity.
        """
        if data.shape[2] == 3:
            return data[:, :, [0, 2]]
        else:
            return data


class AssertTx2(Transform):
    """Assert that the radar data is collected in 3x4 mode."""

    def __call__(
        self, data: Int16[np.ndarray, "T D Tx Rx R"], aug: dict[str, Any] = {},
        idx: int = 0
    ) -> Int16[np.ndarray, "T D Tx Rx R"]:
        """Raises `AssertionError` if the data does not have 3 TX antenna.

        Args:
            data: input 4D data cubes in batch-slow-tx-rx-fast order.

        Returns:
            Always returns identity.
        """
        assert data.shape[2] == 3, "Data was not collected in 3x4 mode."
        return data


class BaseFFT(Transform):
    """FFT base class."""

    @staticmethod
    def _augment(
        data: Complex64[np.ndarray, "T D A ..."], aug: dict[str, Any] = {}
    ) -> Complex64[np.ndarray, "T D A ..."]:
        """Apply radar representation-agnostic data augmentations.

        Args:
            data: input post-FFT data cubes in batch-doppler-azimuth... order.
            aug: data augmentations to apply.

        Returns:
            Data, with azimuth and/or doppler flips applied.
        """
        if aug.get("azimuth_flip"):
            data = np.flip(data, axis=2)
        if aug.get("doppler_flip"):
            data = np.flip(data, axis=1)
        return data


class FFTPrecomputed(BaseFFT):
    """Pre-computed 4D FFT (only handling augmentations).

    Augmentations:

    - `azimuth_flip`: flip along azimuth axis.
    - `doppler_flip`: flip along doppler axis.
    """

    def __call__(
        self, data: Complex64[np.ndarray, "T D A E R"],
        aug: dict[str, Any] = {}, idx: int = 0
    ) -> Complex64[np.ndarray, "T D A E R"]:
        """Apply augmentations.

        Args:
            data: data with 4D fft (i.e. :py:class:`.FFTArray`) pre-computed,
                but no dataloader augmentations applied.
            aug: data augmentations to apply.

        Returns:
            Data, with azimuth and/or doppler flips applied.
        """
        return self._augment(data, aug=aug)


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
        self, data: Complex64[np.ndarray, "T D Tx Rx R"],
        aug: dict[str, Any] = {}, idx: int = 0
    ) -> Complex64[np.ndarray, "T D A R"]:
        """Apply augmentations.

        Args:
            data: raw 4D cubes in time-slow-tx-rx-fast order.
            aug: data augmentations to apply.

        Returns:
            FFT'd cubes with data augmentations applied.
        """
        t, d, tx, rx, r = data.shape
        assert tx == 2, "Only 2-tx mode is supported."

        iq_dar = data.reshape(t, d, tx * rx, r)

        if self.pad > 0:
            zeros = np.zeros([t, d, self.pad, r], dtype=np.complex64)
            iq_dar = np.concatenate([iq_dar, zeros], axis=2)

        dar = fft.fftn(iq_dar, axes=[x + 1 for x in self.axes])
        dar_shf = fft.fftshift(
            dar, axes=[x + 1 for x in self.axes if x in (0, 1)])

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
        self, data: Complex64[np.ndarray, "T D Tx Rx R"],
        aug: dict[str, Any] = {}, idx: int = 0
    ) -> Complex64[np.ndarray, "T D A E R"]:
        """Apply augmentations.

        Args:
            data: raw 4D cubes in time-slow-tx-rx-fast order.
            aug: data augmentations to apply.

        Returns:
            FFT'd cubes with data augmentations applied.
        """
        t, d, tx, rx, r = data.shape

        assert tx == 3, "Only 3-tx mode is supported."
        assert rx == 4, "Only 4-rx mode is supported."
        iq_daer = np.zeros((t, d, 8, 2, r), dtype=np.complex64)
        iq_daer[:, :, 0:4, 0, :] = data[:, :, 0, :, :]
        iq_daer[:, :, 4:8, 0, :] = data[:, :, 2, :, :]
        iq_daer[:, :, 2:6, 1, :] = data[:, :, 1, :, :]

        if self.pad > 0:
            zeros = np.zeros([t, d, self.pad, 2, r], dtype=np.complex64)
            iq_daer = np.concatenate([iq_daer, zeros], axis=2)

        daer = fft.fftn(iq_daer, axes=[x + 1 for x in self.axes])
        daer_shf = fft.fftshift(
            daer, axes=[x + 1 for x in self.axes if x in (0, 1, 2)])

        return self._augment(daer_shf, aug=aug)


class DopplerShuffle(Transform):
    """Shuffle the doppler and/or time axis to destroy all doppler information.

    Used only for ablations, obviously don't do this in practice!

    Args:
        doppler: whether to shuffle the doppler axis.
        time: whether to shuffle the time axis.
    """

    def __init__(self, path: str, doppler=True, time=True) -> None:
        self._doppler = doppler
        self._time = time

    def __call__(
        self, data: Complex64[np.ndarray, "T D Tx Rx R"],
        aug: dict[str, Any] = {}, idx: int = 0
    ) -> Complex64[np.ndarray, "T D Tx Rx R"]:
        rng = np.random.default_rng()
        if self._time:
            rng.shuffle(data, axis=0)
        if self._doppler:
            rng.shuffle(data, axis=1)
        return data
