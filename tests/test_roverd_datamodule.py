"""Regression tests for RoverD datamodule split construction."""

from collections.abc import Sequence
from typing import Any, cast

import nrdk.roverd.dataloader as dataloader_module
from nrdk.roverd.dataloader import datamodule


class _Pipeline:
    def sample(self, data):
        return data

    def collate(self, data):
        return data

    def batch(self, data):
        return data


class _Dataset:
    def __init__(self, traces: Sequence[str]) -> None:
        self.traces = tuple(traces)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {"index": index}

    def __len__(self) -> int:
        return len(self.traces)


def test_explicit_validation_traces_are_used_in_full(monkeypatch) -> None:
    """Use explicit train and validation trace lists without slicing."""
    calls: list[tuple[str, ...]] = []

    def dataset(traces: Sequence[str]) -> _Dataset:
        calls.append(tuple(traces))
        return _Dataset(traces)

    def unexpected_split(*args, **kwargs):
        raise AssertionError("Explicit trace splits must not call roverd.split")

    monkeypatch.setattr(dataloader_module.roverd, "split", unexpected_split)
    module = datamodule(
        dataset=dataset,  # type: ignore[arg-type]
        traces={
            "train": ["train/a", "train/b"],
            "val": ["val/a"],
            "test": ["test/a"],
        },
        transforms=_Pipeline(),
        samples=0,
        num_workers=0,
    )

    datasets = cast(Any, module)._dataset
    train = datasets["train"]()
    val = datasets["val"]()
    test = datasets["test"]()

    assert train.dataset.traces == ("train/a", "train/b")
    assert val.dataset.traces == ("val/a",)
    assert test.dataset.traces == ("test/a",)
    assert train.meta == {"train": True, "split": "train"}
    assert val.meta == {"train": False, "split": "val"}
    assert test.meta == {"train": False, "split": "test"}
    assert calls == [
        ("train/a", "train/b"),
        ("val/a",),
        ("test/a",),
    ]


def test_legacy_frame_split_behavior_is_preserved(monkeypatch) -> None:
    """Keep deriving frame-level validation when no list is supplied."""
    calls: list[tuple[float, float]] = []
    builds: list[tuple[str, ...]] = []

    def dataset(traces: Sequence[str]) -> _Dataset:
        builds.append(tuple(traces))
        return _Dataset(traces)

    def split(dataset: _Dataset, start: float, end: float) -> _Dataset:
        calls.append((start, end))
        return dataset

    monkeypatch.setattr(dataloader_module.roverd, "split", split)
    module = datamodule(
        dataset=dataset,  # type: ignore[arg-type]
        traces={"train": ["train/a"], "test": ["test/a"]},
        transforms=_Pipeline(),
        samples=0,
        num_workers=0,
        ptrain=0.4,
        pval=0.2,
    )

    datasets = cast(Any, module)._dataset
    train = datasets["train"]()
    val = datasets["val"]()
    test = datasets["test"]()

    assert calls == [(0.0, 0.4), (0.8, 1.0)]
    assert builds == [("train/a",), ("test/a",)]
    assert train.meta == {"train": True, "split": "train"}
    assert val.meta == {"train": False, "split": "val"}
    assert test.meta == {"train": False, "split": "test"}
