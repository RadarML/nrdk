"""Dataloader."""

import os, json
import numpy as np
from torch.utils.data import Dataset

from jaxtyping import Num
from beartype.typing import Callable

from .transforms import BaseTransform


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


class RoverTrace:
    """Single rover trace."""

    def __init__(
        self, path: str, indices: str = "_fusion/indices.npz",
        transform: dict[str, list[Callable[[str], BaseTransform]]] = {}
    ) -> None:
        npz = np.load(os.path.join(path, indices))
        self.indices = npz["indices"]
        self.channel_names = {
            n: i for i, n in enumerate(npz["sensors"])}
        self.transform = {
            k: [tf(path) for tf in v] for k, v in transform.items()}

        self.channels = {
            "radar": RawChannel(os.path.join(path, "radar"), "iq"),
            "lidar": RawChannel(os.path.join(path, "_lidar"), "rng")}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        def apply_transform(k, data):# -> Any:
            for tf in self.transform.get(k, []):
                data = tf(data)
            return data

        return {
            k: apply_transform(k, v[self.indices[idx, self.channel_names[k]]])
            for k, v in self.channels.items()}


class RoverData(Dataset):
    """Collection of rover traces.
    
    Parameters
    ----------
    paths: list of dataset paths to include.
    indices: correspondence indices, as a subpath within each dataset.
    transform: transformations to apply to radar, lidar data.
    """

    def __init__(
        self, paths: list[str], indices: str = "_fusion/indices.npz",
        transform: dict[str, list[Callable[[str], BaseTransform]]] = {}
    ) -> None:
        self.traces = [
            RoverTrace(p, transform=transform, indices=indices) for p in paths]

    def __len__(self):
        return sum(len(t) for t in self.traces)

    def __getitem__(self, idx):
        for trace in self.traces:
            if idx < len(trace):
                return trace[idx]
            idx -= len(trace)
        else:
            raise ValueError("Index out of bounds.")
