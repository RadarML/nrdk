"""Train radar model."""

import json
import os
import time
from argparse import ArgumentParser

import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy

from deepradar import DeepRadar, config


def _parse():
    p = ArgumentParser(description="Train radar model.")

    g = p.add_argument_group("Path")
    g.add_argument(
        "-p", "--path", default="data", help="Root dataset directory.")
    g.add_argument(
        "-o", "--out", default="results", help="Root results directory.")

    g = p.add_argument_group("Training")
    g.add_argument(
        "-c", "--cfg", nargs='+', default=None,
        help="Training configuration; see `deepradar.config` for parsing "
        "rules. Must be specified unless resuming with "
        "`--checkpoint <checkpoint>`.")
    g.add_argument(
        "--cfg_dir", default="config", help="Configuration base directory.")
    g.add_argument(
        "-k", "--checkpoint", default=None,
        help="Checkpoint to load, if specified. Should have the structure "
        "`<folder>/checkpoints/<checkpoint>.ckpt`, where `folder` contains a "
        "`hparams.yaml` file.")
    g.add_argument(
        "--epochs", default=-1, type=int, help="Maximum number of epochs.")
    g.add_argument(
        "--patience", default=5, type=int,
        help="Stop after this many validation checks with no improvement.")

    g = p.add_argument_group("Logging")
    g.add_argument(
        "-n", "--name", default=None,
        help="Method name (for experiment tracking only).")
    g.add_argument(
        "-v", "--version", default=None,
        help="Experiment version (for experiment tracking only).")
    g.add_argument(
        "--val_interval", default=0.5, type=float,
        help="Validation interval, as a fraction of each epoch.")
    g.add_argument(
        "--log_example_interval", default=500, type=int,
        help="Interval to log example train images.")
    g.add_argument(
        "--log_interval", default=100, type=int,
        help="Logging interval for training statistics.")
    g.add_argument(
        "--num_checkpoints", default=-1, type=int,
        help="Number of checkpoints to save.")

    return p


def _main(args):
    if args.cfg is None:
        if args.checkpoint is not None:
            experiment_dir = os.path.dirname(os.path.dirname(args.checkpoint))
            args.cfg = [os.path.join(experiment_dir, "hparams.yaml")]
        else:
            print(
                "Must specify a `config.yaml` file if not resuming training.")
            exit(1)
    else:
        args.cfg = [os.path.join(args.cfg_dir, c) for c in args.cfg]

    model_cfg = config.load_config(args.cfg)
    if args.checkpoint is None:
        model = DeepRadar(**model_cfg)
    else:
        model = DeepRadar.load_from_checkpoint(
            args.checkpoint, hparams_file=args.cfg[0])

    # Bypass save_hyperparameters
    model.configure(log_interval=args.log_example_interval, num_examples=16)

    data = model.get_dataset(args.path)

    checkpoint = ModelCheckpoint(
        save_top_k=args.num_checkpoints, monitor="loss/val",
        save_last=True, dirpath=None)
    stopping = EarlyStopping(
        monitor="loss/val", min_delta=0.0,
        patience=args.patience, mode="min")
    logger = TensorBoardLogger(
        args.out, name=args.name, version=args.version,
        default_hp_metric=False)
    strategy = DDPStrategy(find_unused_parameters=True)
    trainer = L.Trainer(
        logger=logger, log_every_n_steps=args.log_interval,
        callbacks=[checkpoint, stopping], max_steps=-1, max_epochs=args.epochs,
        val_check_interval=args.val_interval, strategy=strategy,
        precision="16-mixed")

    start = time.perf_counter()
    trainer.fit(model=model, datamodule=data)
    duration = time.perf_counter() - start

    with open(os.path.join(logger.log_dir, "meta.json"), 'w') as f:
        json.dump({
            "best": os.path.basename(checkpoint.best_model_path),
            "duration": duration
        }, f)


if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')
    _main(_parse().parse_args())
