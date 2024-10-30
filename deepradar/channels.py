"""Different types of data channels."""

import os
from abc import ABC, abstractmethod
from functools import partial

import numpy as np
from beartype.typing import Any, Callable, Optional, Union
from jaxtyping import Num, UInt
from roverd import Dataset

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
        indices: index transformation from trace to channel index; if `None`,
            no transformation is applied.
        transform: list of :py:class:`Transform` to apply to the data.
    """

    def __init__(
        self, dataset: str, indices: Optional[UInt[np.ndarray, "N"]] = None,
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
        if self._indices is None:
            data = self._index(idx)
        else:
            data = self._index(self._indices[idx])

        for tf in self._transforms:
            data = tf(data, aug=aug, idx=int(idx))
        return data

    @classmethod
    def from_config(
        cls, dataset: str, indices: Optional[Num[np.ndarray, "N"]],
        transform: list[dict] = [], **kwargs
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
        self, dataset: str, indices: Optional[UInt[np.ndarray, "N"]],
        sensor: str, channel: str,
        transform: list[Callable[[str], transforms.Transform]] = []
    ) -> None:
        super().__init__(dataset=dataset, indices=indices, transform=transform)
        self.channel = Dataset(dataset)[sensor][channel]

    def _index(self, idx: Index) -> Num[np.ndarray, "..."]:
        try:
            return self.channel.read(int(idx), samples=1)[0]
        except IndexError as e:
            print(self.channel, idx)
            raise(e)


class NPChannel(Channel):
    """N-d time series stream stored in a numpy array.

    NOTE: the numpy array is fully loaded into (main) memory by this loader.

    Args:
        dataset: dataset path.
        path: path of file within dataset.
        keys: keys to load from the `.npz` archive. If `keys` is a str, single
            arrays are yielded instead of dicts of arrays.
        transform: list of :class:`Transform` to apply to the data.
    """

    def __init__(
        self, dataset: str, indices: Optional[UInt[np.ndarray, "N"]],
        path: str, keys: list[str] | str = [],
        transform: list[Callable[[str], transforms.Transform]] = []
    ) -> None:
        super().__init__(dataset=dataset, indices=indices, transform=transform)
        npz = np.load(os.path.join(dataset, path))

        if isinstance(keys, str):
            self.arr = npz[keys]
        else:
            self.arr = {k: npz[k] for k in keys}

        self.index_map: Optional[UInt[np.ndarray, "N2"]] = None
        if "mask" in npz:
            mask = npz["mask"]
            self.index_map = np.zeros(mask.shape, dtype=np.uint32)
            self.index_map[mask] = np.arange(np.sum(mask), dtype=np.uint32)

    def _index(self, idx: Index) -> Any:
        if self.index_map is not None:
            idx = self.index_map[idx]

        if isinstance(self.arr, dict):
            return {k: v[idx] for k, v in self.arr.items()}
        else:
            return self.arr[idx]


class MetaChannel(Channel):
    """Sensor metadata dummy data.

    Always returns `None`.
    """

    def _index(self, idx: Index) -> Any:
        return None
