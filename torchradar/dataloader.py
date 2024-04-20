"""Lightning-based dataloader for a collection of Rover traces."""

import os, json
import multiprocessing
import numpy as np
from functools import partial

from torch.utils.data import Dataset, DataLoader
import lightning as L

from beartype.typing import Union, TypedDict, Callable, Any
from jaxtyping import Num

from . import transforms


#: Any type which can be used as an index
Index = Union[np.integer, int]


#: A Transform is a callable which takes the dataset path.
Transform = Callable[[str], transforms.BaseTransform]


class TransformSpec(TypedDict):
    """Transform specification."""
    name: str
    args: dict[str, Any]


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

    def __getitem__(self, idx: Index) -> Num[np.ndarray, "..."]:
        with open(os.path.join(self.path), 'rb') as f:
            f.seek(self.stride * idx)
            raw = f.read(self.stride)
        return np.frombuffer(raw, dtype=self.dtype).reshape(self.shape)


class RoverTrace:
    """Single rover trace."""

    def __init__(
        self, path: str, indices: str = "_fusion/indices.npz",
        transform: dict[str, list[Transform]] = {}
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

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: Index):
        def apply_transform(k, data):
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
        transform: dict[str, list[Transform]] = {}
    ) -> None:
        self.traces = [
            RoverTrace(p, transform=transform, indices=indices) for p in paths]

    def __len__(self) -> int:
        return sum(len(t) for t in self.traces)

    def __getitem__(self, idx: Index):
        for trace in self.traces:
            if idx < len(trace):
                return trace[idx]
            idx -= len(trace)
        else:
            raise ValueError("Index out of bounds.")



class RoverDataModule(L.LightningDataModule):
    """Rover dataloaders.
    
    Parameters
    ----------
    path: path to directory containing datasets
    train, val: list of traces to use as the train/val splits
    transform: data transformations to perform.
    batch_size: train batch size.
    debug: whether to run in debug mode. When `debug=True`, use `num_workers=0`
        (run dataloaders in main thread) to allow debuggers to work properly;
        otherwise, uses `num_workers=nproc`.
    """

    def __init__(self,
        path: str, train: list[str] = [], val: list[str] = [],
        transform: dict[str, list[Union[Transform, TransformSpec]]] = {},
        batch_size: int = 64, debug: bool = False
    ) -> None:
        super().__init__()
        self.base = path
        self.train = train
        self.val = val
        self.transform = {
            k: [self.as_transform(t) for t in v] for k, v in transform.items()}
        self.batch_size = batch_size
        self.nproc = 0 if debug else multiprocessing.cpu_count()

    @staticmethod
    def as_transform(spec: Union[Transform, TransformSpec]) -> Transform:
        """Deserialize TransformSpec to transform if required."""
        if isinstance(spec, dict):
            return partial(getattr(transforms, spec["name"]), **spec["args"])
        else:
            return spec

    def train_dataloader(self) -> DataLoader:
        ds = RoverData(
            [os.path.join(self.base, t) for t in self.train],
            transform=self.transform)
        return DataLoader(
            ds, batch_size=self.batch_size, shuffle=True, drop_last=True,
            num_workers=self.nproc)

    def val_dataloader(self) -> DataLoader:
        ds = RoverData(
            [os.path.join(self.base, t) for t in self.val],
            transform=self.transform)
        return DataLoader(
            ds, batch_size=self.batch_size, shuffle=False, drop_last=True,
            num_workers=self.nproc)
