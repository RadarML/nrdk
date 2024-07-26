"""Evaluate radar model."""

import os, json
import numpy as np
from argparse import ArgumentParser

import torch

from deepradar import objectives, config


def _parse():
    p = ArgumentParser(description="Evaluate radar model.")

    p.add_argument(
        "-p", "--path", default="data", help="Root dataset directory.")
    p.add_argument("-m", "--model", help="Path to model.")
    p.add_argument(
        "-t", "--traces", nargs='+', help="Traces to evaluate.", default=[])
    p.add_argument("--batch", default=16, help="Evaluation batch size.")

    return p


def _main(args):

    cfg = config.load_config([os.path.join(args.model, "hparams.yaml")])
    try:
        with open(os.path.join(args.model, "meta.json")) as f:
            meta = json.load(f)
            checkpoint = os.path.join(args.model, "checkpoints", meta["best"])
    except FileNotFoundError:
        checkpoint = os.path.join(args.model, "checkpoints", "last.ckpt")

    model = getattr(
        objectives, cfg["objective"]).load_from_checkpoint(checkpoint)
    data = model.get_dataset(args.model)

    if len(args.traces) == 0:
        raise ValueError("Passed empty `-t [--traces]`.")

    if args.traces[0].endswith('.yaml'):
        spec = config.load_config(args.traces)
        args.traces = spec["traces"]

    for i, trace in enumerate(args.traces):
        out = os.path.join(args.model, "eval", trace + ".npz")
        os.makedirs(os.path.dirname(out), exist_ok=True)

        dataloader = data.eval_dataloader(
            os.path.join(args.path, trace), batch_size=args.batch)
        res = model.evaluate(
            dataloader, desc=f"[{i + 1}/{len(args.traces)}] {trace}")
        np.savez(out, **res)


if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')
    _main(_parse().parse_args())
