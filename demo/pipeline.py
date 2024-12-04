"""Data processing pipeline for real-time inference."""

import time
from abc import ABC, abstractmethod
from functools import partial
from queue import Empty, Queue
from threading import Lock, Thread

import numpy as np
import torch
from awr_api.dca_types import RadarFrame
from beartype.typing import Any, Callable
from jaxtyping import Float

from deepradar import DeepRadar, transforms


class RTStream(ABC):
    """Real-time stream processing with statistics and frame dropping."""

    def __init__(self) -> None:
        self._stat_mutex = Lock()
        self._n_total = 0
        self._n_drop = 0
        self._start = time.perf_counter()

    @abstractmethod
    def __call__(self, data: Any) -> Any:
        """Work which is being done."""

    def drop_frames(self, q: Queue) -> Any:
        """Get the most recent item from a queue, and drop all excess items."""
        items = []
        # First item -- fetch blocking
        items.append(q.get())

        # Once we have an item, flush the queue.
        while True:
            try:
                items.append(q.get_nowait())
            except Empty:
                break

        # Last item is the one we care about. Drop everything else.
        with self._stat_mutex:
            self._n_total += len(items)
            self._n_drop += len(items) - 1
        return items[-1]

    def apply(self, q: Queue) -> Queue:
        """Process stream in a thread worker.

        Reads items from a queue, and applies all transforms sequentially;
        `None` is used to denote termination of the stream. The output is put
        in a new Queue, which is returned.
        """
        out: Queue = Queue()

        def worker():
            while True:
                item = self.drop_frames(q)
                if item is None:
                    break
                else:
                    out.put(self(item))

        Thread(target=worker, daemon=True).start()
        return out

    def statistics(self) -> dict[str, float]:
        """Get stream statistics, and reset current counters."""
        with self._stat_mutex:
            duration = time.perf_counter() - self._start
            stats = {
                "received": self._n_total / duration,
                "processed": (self._n_total - self._n_drop) / duration,
                "dropped": self._n_drop / self._n_total}

            self._start = time.perf_counter()
            self._n_total = 0
            self._n_drop = 0

        return stats


class ProcessingStream(RTStream):
    """Real-time stream preprocessing.

    Note that data augmentations are never applied.

    Args:
        dataset: path to dummy metadata-only dataset; can be empty if none of
            the specified transforms require set metadata.
        transform: list of :py:class:`Transform` to apply to the data.
    """

    def __init__(
        self, dataset: str = "",
        transform: list[Callable[[str], transforms.Transform]] = []
    ) -> None:
        super().__init__()
        self._transforms = [tf(dataset) for tf in transform]

    @classmethod
    def from_config(
        cls, dataset: str = "", transform: list[dict] = []
    ) -> "ProcessingStream":
        """Create stream from config.

        Args:
            dataset: path to dataset.
            indices: trace to sensor index conversion.
            transform: list of transformations to apply.
            kwargs: passthrough for channel type-specific configuration.
        """
        return cls(
            dataset=dataset, transform=[
                partial(getattr(transforms, tf["name"]), **tf["args"])
                for tf in transform])

    def __call__(self, data: RadarFrame) -> Any:
        """Apply all transforms."""
        for tf in self._transforms:
            data = tf(data)
        return data


class ModelStream(RTStream):
    """Apply DeepRadar pytorch model to real time stream."""

    def __init__(self, model: DeepRadar) -> None:
        super().__init__()
        self.model = model

    def __call__(
        self, data: Float[np.ndarray, "..."]
    ) -> dict[str, Float[np.ndarray, "..."]]:
        """Apply model."""
        torched = {"radar": torch.from_numpy(data).to('cuda')[None, ...]}
        with torch.no_grad():
            outputs = self.model(torched)

            rendered = {}
            for objective in self.model.objectives:
                rendered.update(objective.render(torched, outputs, gt=False))

        return {k: v[0] for k, v in rendered.items()}
