"""GRT Reference implementation training script."""

import logging
import os
from collections.abc import Mapping
from time import perf_counter
from typing import Any

import hydra
import torch
import yaml
from lightning.pytorch import callbacks
from omegaconf import DictConfig

from nrdk.config import configure_rich_logging
from nrdk.framework import Result

logger = logging.getLogger("train")


def _load_weights(lightningmodule, path: str, rename: Mapping = {}) -> None:
    weights = Result(path).best if os.path.isdir(path) else path
    lightningmodule.load_weights(weights, rename=rename)


def _get_best(trainer) -> dict[str, Any]:
    for callback in trainer.callbacks:
        if isinstance(callback, callbacks.ModelCheckpoint):
            return {
                "best_k": {
                    os.path.basename(k): v.item()
                    for k, v in callback.best_k_models.items()},
                "best": os.path.basename(callback.best_model_path)
            }
    return {}


def _autoscale_batch_size(batch: int) -> int:
    n_gpus = torch.cuda.device_count()
    if n_gpus > 1:
        batch_new = batch // n_gpus
        logger.info(
            f"Auto-scaling batch size by n_gpus={n_gpus}: "
            f"{batch} -> {batch_new}")
        return batch_new
    return batch


@hydra.main(version_base=None, config_path="./config", config_name="default")
def train(cfg: DictConfig) -> None:
    """Train a model using the GRT reference implementation."""
    torch.set_float32_matmul_precision('high')

    _log_level = configure_rich_logging(cfg.meta.get("verbose"))
    logger.debug(f"Configured with log level: {_log_level}")

    if cfg["meta"]["name"] is None or cfg["meta"]["version"] is None:
        logger.error("Must set `meta.name` and `meta.version` in the config.")
        return
    cfg["datamodule"]["batch_size"] = _autoscale_batch_size(
        cfg["datamodule"]["batch_size"])

    def _inst(path, *args, **kwargs):
        return hydra.utils.instantiate(
            cfg[path], _convert_="all", *args, **kwargs)

    transforms = _inst("transforms")
    datamodule = _inst("datamodule", transforms=transforms)
    lightningmodule = _inst("lightningmodule", transforms=transforms)
    trainer = _inst("trainer")
    if "base" in cfg:
        _load_weights(lightningmodule, **cfg['base'])

    if cfg["meta"]["compile"]:
        lightningmodule.compile()

    start = perf_counter()
    logger.info(
        f"Start training @ {cfg["meta"]["results"]}/{cfg["meta"]["name"]}/"
        f"{cfg["meta"]["version"]} [t={start:.3f}]")
    trainer.fit(
        model=lightningmodule, datamodule=datamodule,
        ckpt_path=cfg['meta']['resume'])
    duration = perf_counter() - start
    logger.info(
        f"Training completed in {duration / 60 / 60:.2f}h (={duration:.3f}s).")

    meta: dict[str, Any] = {"duration": duration}
    meta.update(_get_best(trainer))
    logger.info(f"Best checkpoint: {meta.get('best')}")
    meta_path = os.path.join(trainer.logger.log_dir, "checkpoints.yaml")
    with open(meta_path, 'w') as f:
        yaml.dump(meta, f, sort_keys=False)


if __name__ == "__main__":
    train()
