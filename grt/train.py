import os
from time import perf_counter

import hydra
import torch
from lightning.pytorch import callbacks


@hydra.main(version_base=None, config_path="./config", config_name="default")
def train(cfg):

    torch.set_float32_matmul_precision('medium')

    def _inst(path, *args, **kwargs):
        return hydra.utils.instantiate(cfg[path], *args, **kwargs)

    transforms = _inst("transforms")
    datamodule = _inst("datamodule", transforms=transforms)
    lightningmodule = _inst(
        "lightningmodule", model=_inst("model"),
        objective=_inst("objectives"), transforms=transforms)
    trainer = _inst("trainer")

    start = perf_counter()
    trainer.fit(model=lightningmodule, datamodule=datamodule)
    duration = perf_counter() - start

    for callback in trainer.callbacks:
        if isinstance(callback, callbacks.ModelCheckpoint):
            callback.to_yaml(
                os.path.join(trainer.logger.log_dir, "checkpoints.yaml"))


if __name__ == "__main__":
    train()
