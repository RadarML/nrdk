"""Evaluate radar model."""

from argparse import ArgumentParser

import cv2
import numpy as np
import torch
import yaml
from matplotlib import colormaps
from awr_api import AWRSystem
from beartype.typing import cast

from deepradar import DeepRadar
from demo.pipeline import ModelStream, ProcessingStream


def _parse():
    p = ArgumentParser(description="Evaluate radar model.")

    p.add_argument("-m", "--model", help="Path to model.")
    p.add_argument(
        "-k", "--checkpoint", default=None,
        help="Override the default selected checkpoint; should be a file in "
        "{model}/checkpoints/.")
    p.add_argument("-c", "--config", help="Radar configuration (yaml).")

    return p


def _main(args):

    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    radar = AWRSystem(**cfg)

    model = DeepRadar.load_from_experiment(
        args.model, checkpoint=args.checkpoint)
    model_stream = ModelStream(cast(DeepRadar, model))
    preproc_stream = ProcessingStream.from_config(
        transform=model.dataset["channels"]["radar"]["args"]["transform"])

    out_stream = model_stream.apply(
        preproc_stream.apply(radar.qstream(numpy=True)))

    ii = 0
    viridis = (np.array(colormaps['viridis'].colors) * 255).astype(np.uint8)
    while True:
        frame = out_stream.get()
        if frame is None:
            break
        else:
            # quant = ((frame['depth'] / 63) * 255).astype(np.uint8)

            quant = frame['bev']
            img = np.take(viridis, quant, axis=0)
            img = cv2.resize(
                img, (1920, 1920 // 2), interpolation=cv2.INTER_NEAREST)

            cv2.imshow('Depth', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            cv2.waitKey(10)

        ii += 1
        if ii % 100 == 0:
            print("preproc", preproc_stream.statistics())
            print("model", model_stream.statistics())

if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')
    _main(_parse().parse_args())
