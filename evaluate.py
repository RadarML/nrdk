"""Evaluate radar model."""

import json
import os
from argparse import ArgumentParser

import numpy as np
import torch

from deepradar import DeepRadar, config


def _parse():
    p = ArgumentParser(description="Evaluate radar model.")

    p.add_argument(
        "-p", "--path", default="data", help="Root dataset directory.")
    p.add_argument("-m", "--model", help="Path to model.")
    p.add_argument(
        "-t", "--traces", nargs='+', help="Traces to evaluate.",
        default=["eval[indoor,outdoor,bike"])
    p.add_argument(
        "--cfg_dir", default="config", help="Configuration base directory.")
    p.add_argument("--batch", default=16, help="Evaluation batch size.")
    p.add_argument(
        "-k", "--checkpoint", default=None,
        help="Override the default selected checkpoint; should be a file in "
        "{model}/checkpoints/.")

    return p


def _main(args):

    model = DeepRadar.load_from_experiment(args.model, checkpoint=args.checkpoint)
    model = torch.compile(model)
    datamodule = model.get_dataset(args.path)

    if len(args.traces) == 0:
        raise ValueError("Passed empty `-t [--traces]`.")

    traces = config.load_config(
        [os.path.join(args.cfg_dir, t) for t in args.traces])["traces"]
    for i, trace in enumerate(traces):
        out = os.path.join(args.model, "eval", trace + ".npz")
        os.makedirs(os.path.dirname(out), exist_ok=True)

        dataloader = datamodule.eval_dataloader(trace, batch_size=args.batch)
        res = model.evaluate(
            dataloader, desc=f"[{i + 1}/{len(traces)}] {trace}")
        np.savez(out, **res)


if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')
    _main(_parse().parse_args())
