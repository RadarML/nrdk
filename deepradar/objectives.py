"""Radar training objectives."""

import re
import torch
import lightning as L

from beartype.typing import Any
from jaxtyping import Num

import models
from .dataloader import RoverDataModule
from . import metrics
from .utils import polar_to_bev


class BaseModel(L.LightningModule):
    """Composable training objective.

    All hyperparameters should be encapsulated by the constructor of the
    inheriting class. Environment-specific, non-hyperparameters should be
    passed in via methods.

    Args:
        objective: objective name (ignored).
        dataset: dataset specifications.
        model: model name; should be a `nn.Module` in `models`.
        model_args: args to pass to the model.
        optimizer: optimizer to use (from `torch.optim`).
        default_lr: default learning rate to use.
        optimizer_args: per-layer/parameter optimizer arguments; each key is a
            regex, and each value are additional parameters to pass to matching
            parameters.
        warmup: learning rate warmup period.
    """

    def __init__(
        self, objective: str = "RadarHD", dataset: dict[str, Any] = {},
        model: str = "UNet", model_args: dict[str, Any] = {},
        optimizer: str = "AdamW", default_lr: float = 1e-3,
        optimizer_args: dict[str, dict] = {}, warmup: int = 0,
    ) -> None:
        super().__init__()

        self.model = getattr(models, model)(**model_args)

        self.dataset = dataset
        self.optimizer = optimizer
        self.optimizer_args = optimizer_args
        self.warmup = warmup
        self.default_lr = default_lr

        self.save_hyperparameters()

    def get_dataset(self, path: str, debug: bool = False) -> RoverDataModule:
        """Get datamodule.

        Args:
            path: dataset root directory.
            debug: whether to run in debug mode.

        Returns:
            Corresponding `RoverDataModule`.    
        """
        return RoverDataModule(**self.dataset, path=path, debug=debug)

    def configure(self, **kwargs) -> None:
        """Set non-hyperparameter configurations."""
        pass

    def log_image_comparison(
        self, path: str, y_true: Num[torch.Tensor, "batch h w"],
        y_hat: Num[torch.Tensor, "batch h w"], height: int = 512
    ) -> None:
        """Log a batch of images.

        Args:
            path: log path, e.g. `train/examples`
            y_true, y_hat: images to log.
            height: image height.
        """
        bev_true = torch.cat(list(polar_to_bev(y_true, height=height)), dim=1)
        bev_hat = torch.cat(list(polar_to_bev(y_hat, height=height)), dim=1)
        bev_grid = torch.cat([bev_true, bev_hat], dim=0)
        self.logger.experiment.add_image(  # type: ignore
            path, bev_grid[None, :, :].cpu().numpy(),
            self.global_step, dataformats='CHW')

    def configure_optimizers(self):
        """Configure optimizers."""
        subsets = {s: [] for s in self.optimizer_args}
        nomatch = []
        for name, param in self.named_parameters():
            for pattern in self.optimizer_args:
                if re.match(pattern, name):
                    subsets[pattern].append(param)
                    break
            else:
                nomatch.append(param)

        opt =  getattr(torch.optim, self.optimizer)([
            {"params": subsets[k], **self.optimizer_args[k]}
            for k in subsets
        ] + [{"params": nomatch}], lr=self.default_lr)

        if self.warmup > 0:
            return {
                "optimizer": opt,
                "lr_scheduler": {
                    "scheduler": torch.optim.lr_scheduler.LinearLR(
                        opt, start_factor=1e-3, end_factor=1.0,
                        total_iters=self.warmup),
                    "interval": "step"
                }
            }
        else:
            return opt


class RadarHD(BaseModel):
    """Radar -> lidar.
    
    Args:
        bce_weight: BCE loss weight; Dice loss is weighted `1 - bce_weight`.
        kwargs: see `BaseModel`.
    """

    def __init__(self, bce_weight: float = 0.9, **kwargs) -> None:
        super().__init__(**kwargs)
        self.loss = metrics.CombinedDiceBCE(
            bce_weight=bce_weight, range_weighted=True)
        self.metrics = {"chamfer_l2": metrics.Chamfer()}
        self.configure()

    def configure(
        self, log_interval: int = 1, num_examples: int = 6, **kwargs
    ) -> None:
        """Set non-hyperparameter configurations."""
        self.log_interval = log_interval
        self.num_examples = num_examples

    def training_step(self, batch, batch_idx):  # type: ignore
        """Standard lightning training step."""
        y_hat = self.model(batch['radar'])
        y_true = batch['lidar'].to(torch.float32)
        loss = self.loss(y_hat, y_true)

        self.log("train/loss", loss, prog_bar=True)
        if self.global_step % self.log_interval == 0:
            self.log_image_comparison(
                "train/sample", y_true[:self.num_examples],
                y_hat[:self.num_examples].detach())

        return loss

    def validation_step(self, batch, batch_idx):  # type: ignore
        """Standard lightning validation step."""
        if batch_idx == 0:
            samples = self.trainer.datamodule.val_samples  # type: ignore
            radar = torch.Tensor(samples['radar']).to(batch['radar'].device)
            y_true = torch.Tensor(samples['lidar']).to(batch['lidar'].device)
            y_hat = self.model(radar)
            self.log_image_comparison("val/sample", y_true, y_hat)

        y_hat = self.model(batch['radar'])
        y_true = batch['lidar'].to(torch.float32)
        metrics = {
            "val/{}".format(k): v(y_hat > 0.5, batch['lidar'])
            for k, v in self.metrics.items()}
        metrics["val/loss"] = self.loss(y_hat, y_true)
        self.log_dict(metrics)
