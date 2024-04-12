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


class FFT3(BaseTransform):
    """Range-doppler-antenna FFT."""

    def __call__(
        self, data: Complex64[np.ndarray, "D Tx Rx R"]
    ) -> Float32[np.ndarray, "D R F"]:
        iq_dar = data.reshape(data.shape[0], -1, data.shape[-1])
        dar = fft.fftn(iq_dar, axes=(0, 1, 2), )
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
    ) -> UInt[np.ndarray, "Nr Az2"]:
        el, az = data.shape
        crop_el = el // 2 - 2
        crop_az = az // 4
        x_crop = data[crop_el:-crop_el, crop_az:-crop_az]

        bin = (x_crop // (self.resolution * 1000)).astype(np.uint16)
        bin[bin >= self.bins] = 0
        res = np.zeros((self.bins, bin.shape[1]), dtype=bool)
        for i in range(bin.shape[1]):
            res[:, i][bin[:, i]] = True
        res[0, :] = 0
        return res[:-1]


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
