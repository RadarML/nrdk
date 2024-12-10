"""Evaluate radar model."""

from argparse import ArgumentParser

import cv2
import numpy as np
import torch
import yaml
from awr_api import AWRSystem
from beartype.typing import cast
from matplotlib import colormaps

from deepradar import DeepRadar
from demo.pipeline import ModelStream, ProcessingStream


def _parse():
    p = ArgumentParser(description="Evaluate radar model.")

    p.add_argument("-t", "--type", help="Model output type.", default='depth')
    p.add_argument("-m", "--model", help="Path to model.")
    p.add_argument(
        "-k", "--checkpoint", default=None,
        help="Override the default selected checkpoint; should be a file in "
        "{model}/checkpoints/.")
    p.add_argument("-c", "--config", help="Radar configuration (yaml).")
    p.add_argument("-s", "--scale", default=1.0, type=float, help="Colormap scale.")

    return p


def _main(args):

    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    radar = AWRSystem(**cfg)

    model = DeepRadar.load_from_experiment(
        args.model, checkpoint=args.checkpoint)
    model.eval()
    model_stream = ModelStream(cast(DeepRadar, model))
    preproc_stream = ProcessingStream.from_config(
        transform=model.dataset["channels"]["radar"]["args"]["transform"])

    out_stream = model_stream.apply(
        preproc_stream.apply(radar.qstream(numpy=True)))

    if args.type == 'depth':
        cmap = (np.array(colormaps['viridis'].colors) * 255).astype(np.uint8)
    elif args.type == 'bev':
        cmap = (np.array(colormaps['inferno'].colors) * 255).astype(np.uint8)
    elif args.type == 'seg':
        with open("schema/colors.yaml") as f:
            cmap = np.array(yaml.load(f, Loader=yaml.SafeLoader)["colors"], dtype=np.uint8)
    else:
        raise ValueError(
            f"Unknown model type: {args.type} "
            "(must be `depth`, `bev`, or `seg`).")

    window_name = f"grt:{args.type}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(
        window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    ii = 0
    while True:
        frame = out_stream.get()
        if frame is None:
            break
        else:
            if args.type == 'depth':
                quant = (
                    np.clip(frame['depth'] / 63 * args.scale, 0, 1) * 255).astype(np.uint8)
            elif args.type == 'bev':
                quant = frame['bev']
            else:
                quant = frame['seg']

            # quant = frame['bev']
            img = np.take(cmap, quant, axis=0)
            img = cv2.resize(
                img, (1920, 1080), interpolation=cv2.INTER_NEAREST)

            cv2.imshow(window_name, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            cv2.waitKey(10)

        ii += 1
        if ii % 100 == 0:
            print("preproc", preproc_stream.statistics())
            print("model", model_stream.statistics())

if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')
    _main(_parse().parse_args())
