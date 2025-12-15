"""Dataloader creation."""

from collections.abc import Callable, Mapping, Sequence
from functools import cache, partial

import roverd
from abstract_dataloader import spec
from abstract_dataloader.ext.lightning import ADLDataModule
from abstract_dataloader.generic import DatasetMeta


def datamodule(
    dataset: Callable[[Sequence[str]], roverd.Dataset],
    traces: Mapping[str, Sequence[str]],
    transforms: spec.Pipeline,
    batch_size: int = 32, samples: int | Sequence[int] = 0,
    num_workers: int = 32, prefetch_factor: int | None = 2,
    subsample: Mapping[str, int | float | None] = {},
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
        transforms: data preprocessing pipeline to apply.
        batch_size: dataloader batch size.
        samples: number of validation-set samples to prefetch for
            visualizations (or a list of indices to use). Note that these
            samples are always held in memory! Set `samples=0` to disable.
        num_workers: number of worker processes during data loading and
            CPU-side processing.
        prefetch_factor: number of batches to fetch per worker.
        subsample: Sample only a (low-discrepancy) subset of samples on each
            split specified here instead of using all samples.
        ptrain: proportion of the data to use for the training split; takes
            the first `ptrain` of each trace.
        pval: proportion of the data to use for the validation split; takes the
            last `pval` of each trace.

    Returns:
        Fully initialized datamodule with the datasets for each split still
            lazily initialized.
    """
    def test_closure(split) -> Callable[[], DatasetMeta]:
        def closed():
            return DatasetMeta(
                dataset(split), meta={"train": False, "split": split})
        return closed

    splits = {k: test_closure(v) for k, v in traces.items() if k != "train"}

    if "train" in traces:
        train_val = cache(partial(dataset, traces["train"]))

        def train():
            return DatasetMeta(
                roverd.split(train_val(), start=0.0, end=ptrain),
                meta={"train": True, "split": "train"})

        def val():
            return DatasetMeta(
                roverd.split(train_val(), start=1 - pval, end=1.0),
                meta={"train": False, "split": "val"})

        splits["train"] = train
        splits["val"] = val

    return ADLDataModule(
        dataset=splits, transforms=transforms, batch_size=batch_size,
        samples=samples, num_workers=num_workers,
        prefetch_factor=prefetch_factor, subsample=subsample)
