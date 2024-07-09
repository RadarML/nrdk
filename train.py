"""Train radar model."""

import os, yaml
from argparse import ArgumentParser

import lightning as L
import torch

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from deepradar import objectives


def _parse():
    p = ArgumentParser(description="Train radar model.")

    g = p.add_argument_group("Path")
    g.add_argument(
        "-p", "--path", default="data", help="Root dataset directory.")
    g.add_argument(
        "-o", "--out", default="results", help="Root results directory.")

    g = p.add_argument_group("Training")
    g.add_argument(
        "-c", "--cfg", default="config.yaml",
        help="Training configuration. Must be specified unless resuming with "
        "`--checkpoint <checkpoint>`.")
    g.add_argument(
        "-k", "--checkpoint", default=None,
        help="Checkpoint to load, if specified. Should have the structure "
        "`<folder>/checkpoints/<checkpoint>.ckpt`, where `folder` contains a "
        "`hparams.yaml` file.")
    g.add_argument(
        "-s", "--steps", default=-1, type=int, help="Maximum number of steps.")

    g = p.add_argument_group("Logging")
    g.add_argument(
        "-n", "--name", default="default",
        help="Method name (for experiment tracking only).")
    g.add_argument(
        "-v", "--version", default=None,
        help="Experiment version (for experiment tracking only).")
    g.add_argument(
        "--val_interval", default=0.5, type=float,
        help="Validation interval, as a fraction of each epoch.")
    g.add_argument(
        "--log_example_interval", default=200, type=int,
        help="Interval to log example train images.")
    g.add_argument(
        "--log_interval", default=200, type=int,
        help="Logging interval for training statistics.")
    g.add_argument(
        "--num_checkpoints", default=-1, type=int,
        help="Number of checkpoints to save.")

    return p


def _main(args):
    if args.cfg is None:
        if args.checkpoint is not None:
            experiment_dir = os.path.dirname(os.path.dirname(args.checkpoint))
            args.cfg = os.path.join(experiment_dir, "hparams.yaml")
        else:
            print(
                "Must specify a `config.yaml` file if not resuming training.")
            exit(1)

    with open(args.cfg) as f:
        cfg = yaml.safe_load(f)

        if "objective" not in cfg:
            print("Config file must specify an `objective`.")
            exit(1)

    if args.checkpoint is None:
        model = getattr(objectives, cfg["objective"])(**cfg)
    else:
        model = getattr(objectives, cfg["objective"]).load_from_checkpoint(
            args.checkpoint, hparams_file=args.cfg)

    # Bypass save_hyperparameters
    model.configure(log_interval=args.log_example_interval, num_examples=6)

    checkpoint = ModelCheckpoint(
        save_top_k=args.num_checkpoints, monitor="loss/val",
        save_last=True, dirpath=None)
    logger = TensorBoardLogger(
        args.out, name=args.name, version=args.version,
        default_hp_metric=False)

    data = model.get_dataset(args.path)
    trainer = L.Trainer(
        logger=logger, log_every_n_steps=args.log_interval,
        callbacks=[checkpoint], max_epochs=-1, max_steps=args.steps,
        val_check_interval=args.val_interval)
    trainer.fit(model=model, datamodule=data)


if __name__ == '__main__':

    torch.set_float32_matmul_precision('high')
    _main(_parse().parse_args())
