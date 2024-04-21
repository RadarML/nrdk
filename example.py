from radar import RoverDataModule, objectives
import lightning as L
import torch
import os

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

torch.set_float32_matmul_precision('high')

dataset = {
    "path": "/media/tianshu/data/radarhd/data",
    "train": [
        "nsh", "cic1", "cic2", "ghc", "smith", "tepper", "scott",
        "hammerschlag", "scaife", "porter", "carnival1", "carnival2",
        "hamburg"
    ],
    "val": ["cic4", "wean"],
    "transform": {
        "lidar": [
            {"name": "Destagger", "args": {}},
            {"name": "Map2D", "args": {}},
        ],
        "radar": [
            {"name": "AssertTx2", "args": {}},
            {"name": "IIQQtoIQ", "args": {}},
            {"name": "FFTArray", "args": {"pad": 0, "axes": [0, 1, 2, 3]}},
            {"name": "ComplexPhase", "args": {}}
        ]
    },
    "batch_size": 32
}

modelspec = {
    "bce_weight": 0.9,
    "model": "RadarTransformer",
    "model_args": {
        "dim": 512,
        "ff": 2048,
        "heads": 8,
        "dropout": 0.1
    },
    "log_interval": 100,
    "optimizer": "AdamW",
    "default_lr": 1e-3,
    "optimizer_args": {}
}

model = objectives.RadarHD(**modelspec)  # type: ignore
# or
# model = objectives.RadarHD.load_from_checkpoint(
#     'path/to/log/last.ckpt', hparams_file='/path/to/log/hparams.yaml')

data = RoverDataModule(**dataset, debug=False)  # type: ignore

checkpoint = ModelCheckpoint(
    save_top_k=3, monitor="loss/val", save_last=True, dirpath=None)
logger = TensorBoardLogger(
    "results2", name=os.getenv("NAME"), version=os.getenv("VERSION"))

trainer = L.Trainer(
    logger=logger, log_every_n_steps=10, callbacks=[checkpoint])
trainer.fit(model=model, datamodule=data)
