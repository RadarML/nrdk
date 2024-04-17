"""Data transformations."""

import os, json
import numpy as np
from scipy import fft
from jaxtyping import Complex64, Int16, UInt, UInt16, Float32
from beartype.typing import Any
from ouster import client


class BaseTransform:
    """Generic transformation.
    
    Parameters
    ----------
    path: dataset path to load any required metadata.
    """

    def __init__(self, path: str) -> None:
        pass

    def __call__(self, data: Any) -> Any:
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
    """Discard antenna TX2 from data collected in 3x4 mode."""

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


class FFT2Pad(BaseTransform):
    """Padded range-antenna FFT."""

    def __init__(self, path: str, pad: int = 24) -> None:
        self.pad = pad

    def __call__(
        self, data: Complex64[np.ndarray, "D Tx Rx R"]
    ) -> Complex64[np.ndarray, "D R A"]:
        assert data.shape[1] == 2, "Only 2-tx mode is supported."

        iq_dar = data.reshape(data.shape[0], -1, data.shape[-1])
        zeros = np.zeros(
            [data.shape[0], self.pad, data.shape[-1]], dtype=np.complex64)
        iq_pad = np.concatenate([iq_dar, zeros], axis=1)
        dar = fft.fftn(iq_pad, axes=(1, 2))
        dar_shf = fft.fftshift(dar, axes=1)

        return np.swapaxes(dar_shf, 1, 2)


class FFT2(BaseTransform):
    """Range-doppler FFT."""

    def __call__(
        self, data: Complex64[np.ndarray, "D Tx Rx R"]
    ) -> Complex64[np.ndarray, "D R A"]:
        iq_dar = data.reshape(data.shape[0], -1, data.shape[-1])
        dar = fft.fftn(iq_dar, axes=(0, 2))
        dar_shf = fft.fftshift(dar, axes=0)

        return np.swapaxes(dar_shf, 1, 2)


class FFT3(BaseTransform):
    """Range-doppler-antenna FFT."""

    def __call__(
        self, data: Complex64[np.ndarray, "D Tx Rx R"]
    ) -> Float32[np.ndarray, "D R F"]:
        assert data.shape[1] == 2, "Only 2-tx mode is supported."
        iq_dar = data.reshape(data.shape[0], -1, data.shape[-1])
        dar = fft.fftn(iq_dar, axes=(0, 1, 2))
        dar_shf = fft.fftshift(dar, axes=(0, 1))

        dra = np.swapaxes(dar_shf, 1, 2)
        return np.concatenate([np.abs(dra) / 1e6, np.angle(dra)], axis=2)


class Destagger(BaseTransform):
    """Destagger lidar data."""

    def __init__(self, path: str) -> None:
        with open(os.path.join(path, "lidar", "lidar.json")) as f:
            self.metadata = client.SensorInfo(f.read())

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
    ) -> UInt[np.ndarray, "Az2 Nr"]:
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


class Depth(BaseTransform):
    """Cropped depth map."""

    def __call__(
        self, data: UInt16[np.ndarray, "El Az"]
    ) -> Float32[np.ndarray, "El2 Az2"]:
        el, az = data.shape
        crop_el = el // 4
        crop_az = az // 4
        x_crop = data[crop_el:-crop_el, crop_az:-crop_az]
        return x_crop.astype(np.float32) / 1000
