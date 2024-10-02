"""Profile model."""

import os
import time
from argparse import ArgumentParser

import numpy as np
import torch
from jaxtyping import Float, Shaped
from tqdm import tqdm

from deepradar import DeepRadar, config


def _parse():
    p = ArgumentParser(description="Profile radar model throughput.")

    p.add_argument(
        "-p", "--path", default="data", help="Root dataset directory.")
    p.add_argument(
        "-o", "--out", default=None,
        help="Output file; if not specified, the output is discarded.")
    p.add_argument(
        "-t", "--trace", default="outdoor/baum",
        help="Reference trace to use for benchmarking.")
    p.add_argument(
        "-c", "--cfg", nargs='+', default=None,
        help="Training configuration; see `deepradar.config` for parsing "
        "rules.")
    p.add_argument(
        "--cfg_dir", default="config", help="Configuration base directory.")
    p.add_argument(
        "--max_batch", default=64, type=int,
        help="Maximum batch size to test.")
    p.add_argument(
        "--iters", default=100, type=int,
        help="Number of benchmarking iterations (batches) to run.")
    return p


def _benchmark(
    model, batch: dict[str, Shaped[torch.Tensor, "1 ..."]],
    batch_size: int, iters: int
) -> Float[np.ndarray, "iters"]:
    batch = {
        k: torch.tile(v, [batch_size] + [1] * (len(v.shape) -1))
        for k, v in batch.items()}
    t = []
    for _ in tqdm(range(iters), desc=f"batch={batch_size}"):
        start = time.perf_counter()
        _ = model(batch)
        t.append(time.perf_counter() - start)
    return 1 / (np.array(t) / batch_size)


def _main(args):
    cfg = [os.path.join(args.cfg_dir, c) for c in args.cfg]
    model = DeepRadar(**config.load_config(cfg))

    dataset = model.get_dataset(args.path)
    dataloader = dataset.eval_dataloader(args.trace, batch_size=1)
    batch_cpu = next(iter(dataloader))
    batch_gpu = {k: v.to('cuda') for k, v in batch_cpu.items()}
    model_gpu = model.to("cuda")

    batch_size = 1
    throughput = []
    while batch_size <= args.max_batch:
        try:
            throughput.append(_benchmark(
                model_gpu, batch_gpu, batch_size, args.iters))
        except torch.cuda.OutOfMemoryError:
            break
        batch_size *= 2

    avg = [np.mean(t) for t in throughput]
    idxmax = np.argmax(avg)
    print("Maximum throughput:")
    print("{:.1f}+/-{:.1f} samples/sec @ batch={}".format(
        avg[idxmax], 2 * np.std(throughput[idxmax]) / np.sqrt(args.iters),
        2 ** idxmax))

    if args.out is not None:
        np.save(args.out, np.array(throughput))


if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')
    _main(_parse().parse_args())
