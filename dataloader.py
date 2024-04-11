"""Dataloader."""

import os, json
import numpy as np

from jaxtyping import Num, Complex64, UInt
from ouster import client
from torch.utils.data import Dataset


class RawChannel:
    """Generic sensor stream."""

    def __init__(self, path: str, channel: str) -> None:

        with open(os.path.join(path, 'meta.json')) as f:
            cfg = json.load(f)

        assert cfg[channel]['format'] == 'raw'
        self.dtype = np.dtype(cfg[channel]['type'])
        self.shape = cfg[channel]['shape']
        self.stride = np.prod(self.shape) * self.dtype.itemsize
        self.path = os.path.join(path, channel)

    def __getitem__(self, idx: int) -> Num[np.ndarray, "..."]:
        with open(os.path.join(self.path), 'rb') as f:
            f.seek(self.stride * idx)
            raw = f.read(self.stride)
        return np.frombuffer(raw, dtype=self.dtype).reshape(self.shape)

    def __len__(self) -> int:
        return os.stat(self.path).st_size // self.stride


class RadarChannel(RawChannel):
    """Radar I/Q stream, including iiqq destaggering."""

    def __init__(self, path: str) -> None:
        super().__init__(path=os.path.join(path, "radar"), channel="iq")

    def __getitem__(self, idx: int) -> Complex64[np.ndarray, "D Tx Rx R"]:
        iiqq = super().__getitem__(idx)
        shape = [*self.shape[:-1], self.shape[-1] // 2]
        iq = np.zeros(shape, dtype=np.complex64)
        iq[..., 0::2] = 1j * iiqq[..., 0::4] + iiqq[..., 2::4]
        iq[..., 1::2] = 1j * iiqq[..., 1::4] + iiqq[..., 3::4]
        return iq
    

class LidarChannel(RawChannel):
    """Elevation-azimuth depth stream, including lidar destaggering."""

    def __init__(self, path: str) -> None:
        super().__init__(path=os.path.join(path, "_lidar"), channel="rng")
        with open(os.path.join(path, "lidar", "lidar.json")) as f:
            self._metadata = client.SensorInfo(f.read())
        
    def __getitem__(self, idx: int) -> UInt[np.ndarray, "El Az"]:
        raw = super().__getitem__(idx)
        full = client.destagger(self._metadata, raw)
        N_el, N_az = full.shape
        return full[N_el // 8: -N_el // 8, N_az // 4: -N_az // 4]


class RoverTrace:
    """Single rover trace."""

    def __init__(
        self, path: str, indices: str = "_fusion/indices.npz"
    ) -> None:
        npz = np.load(os.path.join(path, indices))
        self.indices = npz["indices"]
        self.channel_names = {
            n: i for i, n in enumerate(npz["sensors"])}

        self.channels = {
            "lidar": LidarChannel(path), "radar": RadarChannel(path)}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        res = {
            k: v[self.indices[idx, self.channel_names[k]]]
            for k, v in self.channels.items()}
        return res


class RoverData(Dataset):
    """Collection of rover traces.
    
    Parameters
    ----------
    paths: list of dataset paths to include.
    indices: correspondence indices, as a subpath within each dataset.
    """

    def __init__(
        self, paths: list[str], indices: str = "_fusion/indices.npz",
    ) -> None:
        self.traces = [RoverTrace(p, indices=indices) for p in paths]

    def __len__(self):
        return sum(len(t) for t in self.traces)

    def __getitem__(self, idx):
        for trace in self.traces:
            if idx < len(trace):
                return trace[idx]
            idx -= len(trace)
        else:
            raise ValueError("Index out of bounds.")
