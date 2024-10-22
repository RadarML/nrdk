"""Evaluate radar model."""

import os
import queue
from argparse import ArgumentParser

import numpy as np
import roverd
import torch

from deepradar import DeepRadar, config


def _parse():
    p = ArgumentParser(description="Evaluate radar model.")

    p.add_argument(
        "-p", "--path", default="data", help="Root dataset directory.")
    p.add_argument("-m", "--model", help="Path to model.")
    p.add_argument(
        "-t", "--traces", nargs='+', help="Traces to evaluate.",
        default=["eval[indoor,outdoor,bike]"])
    p.add_argument(
        "--cfg_dir", default="config", help="Configuration base directory.")
    p.add_argument(
        "-b", "--batch", default=16, type=int, help="Evaluation batch size.")
    p.add_argument(
        "-k", "--checkpoint", default=None,
        help="Override the default selected checkpoint; should be a file in "
        "{model}/checkpoints/.")
    p.add_argument(
        "-r", "--render", default=False, action='store_true',
        help="Render visualizations if specified.")

    return p


def evaluate(model, datamodule, trace, args, desc: str):
    """Evaluate a single trace."""
    out = os.path.join(args.model, "eval", trace + ".npz")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    if args.render:
        outd = roverd.sensors.SensorData(
            os.path.join(args.model, "eval", trace),
            create=True, exist_ok=True)

        queues: dict[str, queue.Queue] = {}
        for objective in model.objectives:
            for name, fmt in objective.RENDER_CHANNELS.items():
                queues[name] = queue.Queue()
                channel = outd.create(name, fmt)
                if isinstance(channel, roverd.channels.LzmaFrameChannel):
                    channel.consume(queues[name], thread=True, batch=0)
                else:
                    channel.consume(queues[name], thread=True)

    else:
        queues = None  # type: ignore

    dataloader = datamodule.eval_dataloader(trace, batch_size=args.batch)
    res = model.evaluate(dataloader, desc=desc, outputs=queues)
    np.savez(out, **res)


def _main(args):

    model = DeepRadar.load_from_experiment(
        args.model, checkpoint=args.checkpoint)
    model = torch.compile(model)
    datamodule = model.get_dataset(args.path)

    if len(args.traces) == 0:
        raise ValueError("Passed empty `-t [--traces]`.")

    traces = config.load_config(
        [os.path.join(args.cfg_dir, t) for t in args.traces])["traces"]
    for i, trace in enumerate(traces):
        evaluate(
            model, datamodule, trace, args, f"[{i + 1}/{len(traces)}] {trace}")


if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')
    _main(_parse().parse_args())
