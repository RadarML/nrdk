"""Composable data transformations."""

import os, json
import numpy as np
from scipy import fft

from jaxtyping import Complex64, Int16, UInt, UInt16, Float32, Bool, PyTree
from beartype.typing import Iterable

from ouster import client


class BaseTransform:
    """Generic transformation.
    
    Parameters
    ----------
    path: dataset path to load any required metadata.
    """

    def __init__(self, path: str) -> None:
        pass

    def __call__(self, data: PyTree) -> PyTree:
        raise NotImplementedError()


class IIQQtoIQ(BaseTransform):
    """Convert IIQQ i16 raw data to I/Q complex64 data."""

    def __call__(
        self, data: Int16[np.ndarray, "D Tx Rx R2"]
    ) -> Complex64[np.ndarray, "D Tx Rx R"]:
        shape = [*data.shape[:-1], data.shape[-1] // 2]
        iq = np.zeros(shape, dtype=np.complex64)
        iq[..., 0::2] = 1j * data[..., 0::4] + data[..., 2::4]
        iq[..., 1::2] = 1j * data[..., 1::4] + data[..., 3::4]
        return iq


class DiscardTX2(BaseTransform):
    """Discard antenna TX2 if data was collected in 3x4 mode."""

    def __call__(
        self, data: Int16[np.ndarray, "D Tx Rx R"]
    ) -> Int16[np.ndarray, "D Tx2 Rx R"]:
        if data.shape[1] == 3:
            return data[:, [0, 2]]
        else:
            return data


class AssertTx2(BaseTransform):
    """Assert that the radar data is collected in 3x4 mode."""

    def __call__(
        self, data: Int16[np.ndarray, "D Tx Rx R"]
    ) -> Int16[np.ndarray, "D Tx Rx R"]:
        assert data.shape[1] == 3, "Data was not collected in 3x4 mode."
        return data


class FFTLinear(BaseTransform):
    """N-dimensional FFT on a linear array.
    
    Parameters
    ----------
    pad: azimuth padding; output has shape (tx * rx + pad).
    axes: axes to apply an FFT to; 0=doppler, 1=azimuth, 2=range.
    """

    def __init__(
        self, path: str, pad: int = 0, axes: Iterable[int] = (0, 1, 2)
    ) -> None:
        self.pad = pad
        self.axes = axes

    def __call__(
        self, data: Complex64[np.ndarray, "D Tx Rx R"]
    ) -> Complex64[np.ndarray, "D A R"]:
        d, tx, rx, r = data.shape
        assert tx == 2, "Only 2-tx mode is supported."

        iq_dar = data.reshape(d, tx * rx, r)

        if self.pad > 0:
            zeros = np.zeros([d, self.pad, r], dtype=np.complex64)
            iq_dar = np.concatenate([iq_dar, zeros], axis=1)
        
        dar = fft.fftn(iq_dar, axes=self.axes)
        dar_shf = fft.fftshift(dar, axes=[x for x in self.axes if x in (0, 1)])
        return dar_shf


class FFTArray(BaseTransform):
    """N-dimensional FFT on a nonlinear array.

    Parameters
    ----------
    pad: azimuth padding; output has shape (tx * rx + pad).
    axes: axes to apply an FFT to; 0=doppler, 1=azimuth, 2=elevation, 3=range.
    """

    def __init__(
        self, path: str, pad: int = 0, axes: Iterable[int] = (0, 1, 2, 3)
    ) -> None:
        self.pad = pad
        self.axes = axes

    def __call__(
        self, data: Complex64[np.ndarray, "D Tx Rx R"]
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
        return daer_shf


class ComplexParts(BaseTransform):
    """Convert complex numbers to (real, imag) along a new axis."""

    def __call__(
        self, data: Complex64[np.ndarray, "..."]
    ) -> Float32[np.ndarray, "... 2"]:
        return np.stack([np.real(data), np.imag(data)], axis=-1) / 1e6


class ComplexAmplitude(BaseTransform):
    """Convert complex numbers to amplitude-only."""

    def __call__(
        self, data: Complex64[np.ndarray, "..."]
    ) -> Float32[np.ndarray, "..."]:
        return np.sqrt(np.abs(data)) / 1e3


class ComplexPhase(BaseTransform):
    """Convert complex numbers to (amplitude, phase) along a new axis."""

    def __call__(
        self, data: Complex64[np.ndarray, "..."]
    ) -> Float32[np.ndarray, "... 2"]:
        return np.stack([np.sqrt(np.abs(data)) / 1e3, np.angle(data)], axis=-1)


class Destagger(BaseTransform):
    """Destagger lidar data."""

    def __init__(self, path: str) -> None:
        # ouster-sdk is a naughty, noisy library
        # it is in fact so noisy, that we have cut it off at the os level...
        stdout = os.dup(1)
        os.close(1)
        with open(os.path.join(path, "lidar", "lidar.json")) as f:
            self.metadata = client.SensorInfo(f.read())
        os.dup2(stdout, 1)
        os.close(stdout)

    def __call__(
        self, data: UInt[np.ndarray, "El Az"]
    ) -> UInt[np.ndarray, "El Az"]:
        return client.destagger(self.metadata, data)


class Map2D(BaseTransform):
    """2D lidar map."""

    def __init__(self, path: str) -> None:
        with open(os.path.join(path, "radar", "radar.json")) as f:
            self.resolution: float = json.load(f)["range_resolution"]
        with open(os.path.join(path, "radar", "meta.json")) as f:
            self.bins: int = json.load(f)["iq"]["shape"][-1] // 2

    def __call__(
        self, data: UInt16[np.ndarray, "El Az"]
    ) -> Bool[np.ndarray, "Az2 Nr"]:
        # Crop, convert mm -> m
        el, az = data.shape
        crop_el = el // 4
        crop_az = az // 4
        x_crop = data[crop_el:-crop_el, crop_az:-crop_az] / 1000

        # Project to polar
        el_angles = np.linspace(
            -np.pi / 2, np.pi / 2, x_crop.shape[0], dtype=np.float32)
        z = np.sin(el_angles)[:, None] * x_crop
        r = np.cos(el_angles)[:, None] * x_crop

        # Crop to (-30cm, 0cm)
        r[(z > 0.0) | (z < -0.3)] = 0

        # Create map
        bin = (r // (self.resolution)).astype(np.uint16)
        bin[bin >= self.bins] = 0
        res = np.zeros((bin.shape[1], self.bins), dtype=bool)
        for i in range(bin.shape[1]):
            res[i][bin[:, i]] = True
        res[:, 0] = 0
        return res


class DecimateMap(BaseTransform):
    """Downsample lidar map."""

    def __init__(self, path: str, azimuth: int = 1, range: int = 1) -> None:
        self.azimuth = azimuth
        self.range = range

    def __call__(
        self, data: Bool[np.ndarray, "Az Nr"]
    ) -> Bool[np.ndarray, "Az_dec Nr_dec"]:
        na, nr = data.shape
        return np.any(data.reshape(
            na // self.azimuth, self.azimuth, nr // self.range, self.range
        ), axis=(1, 3))


class Depth(BaseTransform):
    """Cropped depth map, in meters."""

    def __call__(
        self, data: UInt16[np.ndarray, "El Az"]
    ) -> Float32[np.ndarray, "El2 Az2"]:
        el, az = data.shape
        crop_el = el // 4
        crop_az = az // 4
        x_crop = data[crop_el:-crop_el, crop_az:-crop_az]
        return x_crop.astype(np.float32) / 1000
