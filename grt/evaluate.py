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
from omegaconf import DictConfig
from roverd.channels.utils import Prefetch
from roverd.sensors import DynamicSensor
from tqdm import tqdm

from nrdk.framework import Result


def _get_dataloaders(
    cfg: DictConfig, data_root: str, transforms: Any,
    traces: list[str] | None = None, filter: str | None = None,
) -> dict[str, Callable[[], torch.utils.data.DataLoader]]:
    datamodule = hydra.utils.instantiate(
        cfg["datamodule"], transforms=transforms)

    if traces is None and filter is None:
        return {"sample": lambda: datamodule.test_dataloader()}
    else:
        dataset_constructor = hydra.utils.instantiate(
            cfg["datamodule"]["dataset"])
        if traces is None:
            assert filter is not None
            traces_inst = [
                os.path.relpath(t, cfg["meta"]["dataset"])
                for t in hydra.utils.instantiate(
                    cfg["datamodule"]["traces"]["test"])]
            traces = [t for t in traces_inst if re.match(filter, t)]

        def construct(t: str) -> torch.utils.data.DataLoader:
            dataset = dataset_constructor(paths=[t])
            return datamodule.dataloader(dataset, mode="test")

        return {
            t: partial(construct, os.path.join(data_root, t)) for t in traces}


def evaluate(
    path: str, /, sample: int | None = None,
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

    !!! tip

        See [`Result`][nrdk.framework.Result] for details about the expected
        structure of the results directory.

    !!! warning

        Only supports using a single GPU; if multiple GPUs are available,
        use parallel evaluation instead.

    Args:
        path: path to results directory.
        sample: TODO
        traces: TODO
        filter: TODO
        data_root: root dataset directory; if `None`, use the path specified
            in `meta/dataset` in the config.
        device: device to use for evaluation.
        batch: batch size.
        workers: number of workers for data loading.
        prefetch: number of batches to prefetch per worker.
    """
    result = Result(path)
    cfg = result.config()
    if sample is not None:
        cfg["datamodule"]["subsample"]["test"] = sample

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

    transforms = hydra.utils.instantiate(cfg["transforms"])
    lightningmodule = hydra.utils.instantiate(
        cfg["lightningmodule"], transforms=transforms).to(device)
    lightningmodule.load_weights(result.best)

    dataloaders = _get_dataloaders(
        cfg, data_root, transforms, traces=traces, filter=filter)

    def collect_metadata(y_true):
        return {
            f"meta/{k}/ts": getattr(v, "timestamps")
            for k, v in y_true.items()
        }

    for trace, dl_constructor in dataloaders.items():
        dataloader = dl_constructor()
        eval_stream = tqdm(
            Prefetch(lightningmodule.evaluate(
                dataloader, metadata=collect_metadata, device=device)),
            total=len(dataloader), desc=trace)

        output_container = DynamicSensor(
            os.path.join(result.path, "eval", trace),
            create=True, exist_ok=True)
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
            os.path.join(result.path, "eval", trace, "metrics.npz"),
            **metrics, allow_pickle=False)

        output_container.create("ts", meta={
            "format": "raw", "type": "f8", "shape": (),
            "desc": "reference timestamps"}
        ).write(metrics["meta/spectrum/ts"])


if __name__ == "__main__":
    tyro.cli(evaluate)
