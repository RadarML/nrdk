"""GRT Reference implementation training script."""

import os
from time import perf_counter
from typing import Any

import hydra
import torch
import yaml
from lightning.pytorch import callbacks


@hydra.main(version_base=None, config_path="./config", config_name="default")
def train(cfg):
    """Train a model using the GRT reference implementation."""
    torch.set_float32_matmul_precision('medium')

    def _inst(path, *args, **kwargs):
        return hydra.utils.instantiate(
            cfg[path], _convert_="all", *args, **kwargs)

    transforms = _inst("transforms")
    datamodule = _inst("datamodule", transforms=transforms)
    lightningmodule = _inst("lightningmodule", transforms=transforms)
    trainer = _inst("trainer")

    start = perf_counter()
    trainer.fit(model=lightningmodule, datamodule=datamodule)
    duration = perf_counter() - start

    meta: dict[str, Any] = {"duration": duration}
    for callback in trainer.callbacks:
        if isinstance(callback, callbacks.ModelCheckpoint):
            meta["best_k"] = {
                os.path.basename(k): v.item()
                for k, v in callback.best_k_models.items()}
            meta["best"] = os.path.basename(callback.best_model_path)
            break

    meta_path = os.path.join(trainer.logger.log_dir, "checkpoints.yaml")
    with open(meta_path, 'w') as f:
        yaml.dump(meta, f, sort_keys=False)


if __name__ == "__main__":
    train()
