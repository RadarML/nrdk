"""Dataloader creation."""

from collections.abc import Mapping, Sequence
from functools import cache, partial
from typing import Any, Callable

import numpy as np
import roverd
from abstract_dataloader import spec
from abstract_dataloader.ext.lightning import ADLDataModule


class _DatasetMeta(spec.Dataset):
    def __init__(
        self, dataset: spec.Dataset[dict[str, Any]], meta: Any
    ) -> None:
        self.dataset = dataset
        self.meta = meta

    def __getitem__(self, index: int | np.integer) -> dict[str, Any]:
        return {"meta": self.meta, **self.dataset[index]}

    def __len__(self) -> int:
        return len(self.dataset)


def roverd_datamodule(
    dataset: Callable[[Sequence[str]], roverd.Dataset],
    traces: Mapping[str, Sequence[str]],
    datamodule: Callable[
        [Mapping[str, Callable[[], spec.Dataset] | spec.Dataset]],
        ADLDataModule],
    ptrain: float = 0.8, pval: float = 0.2
) -> ADLDataModule:
    """Create a datamodule for a [`roverd` dataset][roverd.Dataset].

    The train split is further split into a separate train and validation split
    using [`roverd.split`][roverd.split].

    !!! info

        The dataset is required to return a `dict[str, ...]`; the train
        split will have an extra `meta = {"train": True}` key added, while
        other splits will have `meta = {"train": False}`.

    Args:
        dataset: dataset constructor with all but the trace names bound.
        traces: trace names to use for each split.
        datamodule: datamodule constructor with all but the datasets for each
            split bound.
        ptrain: proportion of the data to use for the training split; takes
            the first `ptrain` of each trace.
        pval: proportion of the data to use for the validation split; takes the
            last `pval` of each trace.

    Returns:
        Fully initialized datamodule with the datasets for each split still
            lazily initialized.
    """
    def test_closure(split) -> Callable[[], _DatasetMeta]:
        def closed():
            return _DatasetMeta(
                dataset(split), meta={"train": False, "split": split})
        return closed

    splits = {k: test_closure(v) for k, v in traces.items() if k != "train"}

    if "train" in traces:
        train_val = cache(partial(dataset, traces["train"]))

        def train():
            return _DatasetMeta(
                roverd.split(train_val(), start=1.0 - pval, end=1.0),
                meta={"train": True, "split": "train"})

        def val():
            return _DatasetMeta(
                roverd.split(train_val(), start=ptrain, end=1.0),
                meta={"train": False, "split": "val"})

        splits["train"] = train
        splits["val"] = val

    return datamodule(splits)
