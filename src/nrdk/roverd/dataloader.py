"""Dataloader creation."""

import logging
import os
from collections.abc import Callable, Mapping, Sequence
from functools import cache, partial
from typing import Literal

import roverd
from abstract_dataloader import spec
from abstract_dataloader.ext.lightning import ADLDataModule
from abstract_dataloader.generic import DatasetMeta

LoaderProfile = Literal["manual", "safe_spawn", "debug"]
logger = logging.getLogger(__name__)


def _positive_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    return parsed if parsed > 0 else None

def _slurm_cpu_budget() -> int | None:
    for key in ("SLURM_CPUS_PER_GPU", "SLURM_CPUS_PER_TASK"):
        budget = _positive_int(os.getenv(key))
        if budget is not None:
            return budget
    return None

def _resolve_loader_options(
    profile: LoaderProfile,
    num_workers: int,
) -> tuple[int, int | None, str | None, bool]:
    '''
    Resolve dataloader options based on the selected profile.
    
    Returns:
        A tuple of (num_workers, prefetch_factor, multiprocessing_context,
        persistent_workers)
    '''
    if profile == "debug":
        return 0, None, None, False

    if profile == "safe_spawn":
        budget = _slurm_cpu_budget()
        requested_workers = num_workers

        if num_workers < 0:
            # Conservative fallback when no explicit worker count is provided.
            num_workers = 8 if budget is None else budget
            logger.info(
                "Auto-selected num_workers=%s for loader_profile=safe_spawn",
                num_workers,
            )
        if budget is not None and num_workers > budget:
            logger.warning(
                "Capping num_workers from %s to %s based on CPU budget",
                num_workers,
                budget,
            )
            num_workers = budget
        if num_workers < 0:
            raise ValueError(
                f"Invalid num_workers={requested_workers}; expected >=0 or -1.")

        if num_workers == 0:
            return 0, None, None, False

        return (num_workers, 1, "spawn", True)

    if profile != "manual":
        raise ValueError(
            f"Unknown loader_profile={profile!r}; expected one of "
            "'manual', 'safe_spawn', or 'debug'.")
    if num_workers < 0:
        raise ValueError(
            "num_workers must be >=0 when loader_profile='manual'. "
            "Use loader_profile='safe_spawn' with num_workers=-1 for auto.")
    if num_workers == 0:
        return 0, None, None, False
    return num_workers, 2, None, False


def datamodule(
    dataset: Callable[[Sequence[str]], roverd.Dataset],
    traces: Mapping[str, Sequence[str]],
    transforms: spec.Pipeline,
    batch_size: int = 32, samples: int | Sequence[int] = 0,
    num_workers: int = 32,
    loader_profile: LoaderProfile = "manual",
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
        loader_profile: dataloader behavior preset.
            - `manual`: use `fork`-style workers with prefetch=2.
            - `safe_spawn`: use `spawn` + persistent workers + prefetch=1,
              and a conservative auto mode when `num_workers=-1`.
            - `debug`: force single-process loading (`num_workers=0`).
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

    num_workers, prefetch_factor, multiprocessing_context, persistent_workers = _resolve_loader_options(
        profile=loader_profile,
        num_workers=num_workers,
    )
    return ADLDataModule(
        dataset=splits, transforms=transforms, batch_size=batch_size,
        samples=samples, num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        multiprocessing_context=multiprocessing_context,
        persistent_workers=persistent_workers,
        subsample=subsample)
