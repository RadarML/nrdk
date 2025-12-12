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
from rich.logging import RichHandler

from nrdk.framework import Result

logger = logging.getLogger("train")

def _configure_logging(cfg: DictConfig) -> None:
    log_level = cfg.meta.get("verbose", logging.INFO)
    root = logging.getLogger()
    root.setLevel(log_level)
    root.handlers.clear()

    rich_handler = RichHandler(markup=True)
    rich_handler.setFormatter(logging.Formatter(
        "[orange1]%(name)s:[/orange1] %(message)s"))
    root.addHandler(rich_handler)

    logger.debug(f"Configured with log level: {log_level}")


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


@hydra.main(version_base=None, config_path="./config", config_name="default")
def train(cfg: DictConfig) -> None:
    """Train a model using the GRT reference implementation."""
    torch.set_float32_matmul_precision('high')
    _configure_logging(cfg)

    if cfg["meta"]["name"] is None or cfg["meta"]["version"] is None:
        logger.error("Must set `meta.name` and `meta.version` in the config.")
        return

    def _inst(path, *args, **kwargs):
        return hydra.utils.instantiate(
            cfg[path], _convert_="all", *args, **kwargs)

    n_gpus = torch.cuda.device_count()
    if "batch_size" in cfg["datamodule"] and n_gpus > 1:
        batch_new = cfg["datamodule"]["batch_size"] // n_gpus
        logger.info(
            f"Auto-scaling batch size by n_gpus={n_gpus}: "
            f"{cfg["datamodule"]["batch_size"]} -> {batch_new}")
        cfg["datamodule"]["batch_size"] = batch_new

    transforms = _inst("transforms")
    datamodule = _inst("datamodule", transforms=transforms)
    lightningmodule = _inst("lightningmodule", transforms=transforms)
    trainer = _inst("trainer")
    if "base" in cfg:
        _load_weights(lightningmodule, **cfg['base'])

    if cfg["meta"]["compile"]:
        jt_disable = os.environ.get("JAXTYPING_DISABLE", "0").lower()
        if jt_disable not in ("1", "true"):
            logger.error(
                "torch.compile is currently incompatible with jaxtyping; "
                "if you see type errors, set the environment variable "
                "`JAXTYPING_DISABLE=1` to disable jaxtyping checks.")
        lightningmodule = torch.compile(lightningmodule)
        logger.info("LightningModule compiled with torch.compile.")

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
