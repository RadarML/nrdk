"""Different types of data channels."""

import os
import json
from functools import partial
from abc import ABC, abstractmethod
import numpy as np
from beartype.typing import Union, Callable, Any
from jaxtyping import Num, UInt

from . import transforms


#: Any type which can be used as an index
Index = Union[np.integer, int]


class Channel(ABC):
    """Sensor stream.

    Channels must implement `_index`, which indexes into the channel by its
    native index; index alignments and data transformations are handled by the
    base class, and data transformations.

    NOTE: random accesses (e.g. `idx` in arbitrary order) must be supported.

    Args:
        dataset: path to dataset.
        transform: list of :py:class:`Transform` to apply to the data.
        indices: index transformation from trace to channel index.
    """

    def __init__(
        self, dataset: str, indices: UInt[np.ndarray, "N"],
        transform: list[Callable[[str], transforms.Transform]] = [],
    ) -> None:
        self._transforms = [tf(dataset) for tf in transform]
        self._indices = indices

    @abstractmethod
    def _index(self, idx: Index) -> Any:
        pass

    def index(self, idx: Index, aug: dict[str, Any] = {}) -> Any:
        """Index into sensor stream.

        Args:
            idx: trace index. The caller should guarantee that it is in bounds
                for this channel.
            aug: data augmentations to apply; is passsed to each
                :py:class:`Transform`, which is responsible for picking out
                relevant keys.

        Returns:
            Loaded and transformed data corresponding to the global `idx`.
        """
        data = self._index(self._indices[idx])
        for tf in self._transforms:
            data = tf(data, aug=aug)
        return data

    @classmethod
    def from_config(
        cls, dataset: str, indices: Num[np.ndarray, "N"],
        transform: list[transforms.TransformSpec] = [], **kwargs
    ) -> "Channel":
        """Create channel from config.

        Args:
            dataset: path to dataset.
            indices: trace to sensor index conversion.
            transform: list of transformations to apply.
            kwargs: passthrough for channel type-specific configuration.
        """
        return cls(
            dataset=dataset, indices=indices, transform=[
                partial(getattr(transforms, tf["name"]), **tf["args"])
                for tf in transform],
            **kwargs)


class RawChannel(Channel):
    """Generic N-d time series stream in `red-rover` format.

    Args:
        dataset: dataset path.
        sensor: sensor name.
        channel: channel within sensor.
        transform: list of :class:`Transform` to apply to the data.
    """

    def __init__(
        self, dataset: str, indices: UInt[np.ndarray, "N"],
        sensor: str, channel: str,
        transform: list[Callable[[str], transforms.Transform]] = []
    ) -> None:
        super().__init__(dataset=dataset, indices=indices, transform=transform)

        with open(os.path.join(dataset, sensor, 'meta.json')) as f:
            cfg = json.load(f)

        assert cfg[channel]['format'] == 'raw'
        self.dtype = np.dtype(cfg[channel]['type'])
        self.shape = cfg[channel]['shape']
        self.stride = np.prod(self.shape) * self.dtype.itemsize
        self.path = os.path.join(dataset, sensor, channel)

    def _index(self, idx: Index) -> Num[np.ndarray, "..."]:
        with open(os.path.join(self.path), 'rb') as f:
            f.seek(self.stride * idx)
            raw = f.read(self.stride)
        return np.frombuffer(raw, dtype=self.dtype).reshape(self.shape)


class NPChannel(Channel):
    """N-d time series stream stored in a numpy array.

    NOTE: the numpy array is fully loaded into (main) memory by this loader.

    Args:
        dataset: dataset path.
        path: path of file within dataset.
        keys: keys to load from the `.npz` archive.
        transform: list of :class:`Transform` to apply to the data.
    """

    def __init__(
        self, dataset: str, indices: UInt[np.ndarray, "N"], path: str,
        keys: list[str] = [],
        transform: list[Callable[[str], transforms.Transform]] = []
    ) -> None:
        super().__init__(dataset=dataset, indices=indices, transform=transform)
        npz = np.load(os.path.join(dataset, path))
        self.arr = {k: npz[k] for k in keys}

    def _index(self, idx: Index) -> Any:
        return {k: v[idx] for k, v in self.arr.items()}


class MetaChannel(Channel):
    """Sensor metadata dummy data.

    Always returns `None`.
    """

    def _index(self, idx: Index) -> Any:
        return None
