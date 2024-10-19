"""Render radar model."""

import os
import queue
from argparse import ArgumentParser

import roverd
import torch

from deepradar import DeepRadar, config


def _parse():
    p = ArgumentParser(description="Render radar model.")

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

    return p


def _main(args):

    model = DeepRadar.load_from_experiment(
        args.model, checkpoint=args.checkpoint)
    model = torch.compile(model)
    datamodule = model.get_dataset(args.path)

    if len(args.traces) == 0:
        raise ValueError("Passed empty `-t [--traces]`.")

    traces = config.load_config(
        [os.path.join(args.cfg_dir, t) for t in args.traces])["traces"]

    print("Rendering Channels:")
    for i, trace in enumerate(traces):

        out = roverd.sensors.SensorData(
            os.path.join(args.model, "eval", trace),
            create=True, exist_ok=True)

        queues = {}
        for objective in model.objectives:
            for name, fmt in objective.RENDER_CHANNELS.items():
                queues[name] = queue.Queue()
                channel = out.create(name, fmt)
                if isinstance(channel, roverd.channels.LzmaFrameChannel):
                    channel.consume(queues[name], thread=True, batch=0)
                else:
                    channel.consume(queues[name], thread=True)
                if i == 0:
                    print(f"{name:12} {fmt['desc']}")

        dataloader = datamodule.eval_dataloader(trace, batch_size=args.batch)
        model.render(
            dataloader, desc=f"[{i + 1}/{len(traces)}] {trace}",
            outputs=queues)


if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')
    _main(_parse().parse_args())
