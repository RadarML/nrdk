"""GRT reference implementation evaluation script."""

import os
import re
from functools import partial
from queue import Queue
from typing import Any, Callable

import hydra
import numpy as np
import torch
import tyro
import wadler_lindig as wl
from abstract_dataloader import spec
from omegaconf import DictConfig
from roverd.channels.utils import Prefetch
from roverd.sensors import DynamicSensor
from tqdm import tqdm

from nrdk.framework import Result


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


def _get_dataloaders(
    cfg: DictConfig, data_root: str, transforms: Any,
    traces: list[str] | None = None, filter: str | None = None,
    sample: int | None = None
) -> dict[str, Callable[[], torch.utils.data.DataLoader]]:
    datamodule = hydra.utils.instantiate(
        cfg["datamodule"], transforms=transforms)

    if traces is None and filter is None and sample is not None:
        return {"sample": lambda: datamodule.test_dataloader()}
    else:
        dataset_constructor = hydra.utils.instantiate(
            cfg["datamodule"]["dataset"])
        if traces is None:
            traces = [
                os.path.relpath(t, cfg["meta"]["dataset"])
                for t in hydra.utils.instantiate(
                    cfg["datamodule"]["traces"]["test"])]

        _unfiltered = traces
        if filter is not None:
            traces = [t for t in traces if re.match(filter, t)]
        if len(traces) == 0:
            raise ValueError(
                f"No traces match the filter {filter}:\n"
                f"{wl.pprint(_unfiltered)}")

        def construct(t: str) -> torch.utils.data.DataLoader:
            dataset = _DatasetMeta(
                dataset_constructor(paths=[t]),
                meta={"train": False, "split": "test"})

            return datamodule.dataloader(dataset, mode="test")

        return {
            t: partial(construct, os.path.join(data_root, t)) for t in traces}


def evaluate(
    path: str, /, output: str | None = None, sample: int | None = None,
    traces: list[str] | None = None, filter: str | None = None,
    data_root: str | None = None,
    device: str = "cuda:0",
    batch: int = 32, workers: int = 32, prefetch: int = 2
) -> None:
    """Evaluate a trained model.

    Supports three evaluation modes, in order of precedence:

    1. Enumerated traces: evaluate all traces specified by `--trace`, relative
        to the `--data-root`.
    2. Filtered evaluation: evaluate all traces in the configuration
        (`datamodule/traces/test`) that match the provided `--filter` regex.
    3. Sample evaluation: evaluate a pseudo-random `--sample` taken from
        the test set specified in the configuration.

    If none of `--trace`, `--filter`, or `--sample` are provided, defaults to
    evaluating all traces specified in the configuration.

    !!! tip

        See [`Result`][nrdk.framework.Result] for details about the expected
        structure of the results directory.

    !!! warning

        Only supports using a single GPU; if multiple GPUs are available,
        use parallel evaluation instead.

    Args:
        path: path to results directory.
        output: if specified, write results to this directory instead.
        sample: number of samples to evaluate.
        traces: explicit list of traces to evaluate.
        filter: evaluate all traces matching this regex.
        data_root: root dataset directory; if `None`, use the path specified
            in `meta/dataset` in the config.
        device: device to use for evaluation.
        batch: batch size.
        workers: number of workers for data loading.
        prefetch: number of batches to prefetch per worker.
    """
    torch.set_float32_matmul_precision('high')

    result = Result(path)
    cfg = result.config()
    if sample is not None:
        cfg["datamodule"]["subsample"]["test"] = sample

    if output is None:
        output = os.path.join(path, "eval")

    if data_root is None:
        data_root = cfg["meta"]["dataset"]
        if data_root is None:
            raise ValueError(
                "`--data_root` must be specified if `meta/dataset` is not set "
                "in the config.")
    else:
        cfg["meta"]["dataset"] = data_root

    cfg["datamodule"]["batch_size"] = batch
    cfg["datamodule"]["num_workers"] = workers
    cfg["datamodule"]["prefetch_factor"] = prefetch
    cfg["lightningmodule"]["compile"] = False

    transforms = hydra.utils.instantiate(cfg["transforms"])
    lightningmodule = hydra.utils.instantiate(
        cfg["lightningmodule"], transforms=transforms).to(device)
    lightningmodule.load_weights(result.best)

    dataloaders = _get_dataloaders(
        cfg, data_root, transforms,
        traces=traces, filter=filter, sample=sample)

    def collect_metadata(y_true):
        return {
            f"meta/{k}/ts": getattr(v, "timestamps")
            for k, v in y_true.items() if hasattr(v, "timestamps")
        }

    for trace, dl_constructor in dataloaders.items():
        dataloader = dl_constructor()
        eval_stream = tqdm(
            Prefetch(lightningmodule.evaluate(
                dataloader, metadata=collect_metadata, device=device)),
            total=len(dataloader), desc=trace)

        output_container = DynamicSensor(
            os.path.join(output, trace), create=True, exist_ok=True)
        metrics = []
        outputs = {}
        for batch_metrics, vis in eval_stream:
            if len(outputs) == 0:
                for k, v in vis.items():
                    outputs[k] = Queue()
                    output_container.create(
                        k.split("/")[-1], meta={
                            "format": "lzmaf",
                            "type": f"{v.dtype.kind}{v.dtype.itemsize}",
                            "shape": v.shape[1:],
                            "desc": f"eval_render:{k}"
                        }
                    ).consume(outputs[k], thread=True)

            for k, v in vis.items():
                for sample in v:
                    outputs[k].put(sample)
            metrics.append(batch_metrics)

        for q in outputs.values():
            q.put(None)

        metrics = {
            k: np.concatenate([m[k] for m in metrics], axis=0)
            for k in metrics[0]}
        np.savez_compressed(
            os.path.join(output, trace, "metrics.npz"),
            **metrics, allow_pickle=False)

        output_container.create("ts", meta={
            "format": "raw", "type": "f8", "shape": (),
            "desc": "reference timestamps"}
        ).write(metrics["meta/spectrum/ts"])


if __name__ == "__main__":
    tyro.cli(evaluate)
