"""Radar training objectives."""

import re
import torch
from torch import nn, Tensor
from torchvision import transforms, utils
import lightning as L

from beartype.typing import Any
from jaxtyping import Float, Num

import models
from .dataloader import RoverDataModule


class BaseModel(L.LightningModule):
    """Composable training objective.
    
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
        optimizer_args: dict[str, dict] = {}, warmup: int = 0
    ) -> None:
        super().__init__()

        self.model = getattr(models, model)(**model_args)

        self.log_interval = 1
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

    def add_image(
        self, path: str, img: Num[torch.Tensor, "n h w"],
        size: tuple[int, int] = (256, 1024), columns: int = 8
    ) -> None:
        """Log a batch of images.
        
        Args:
            path: log path, e.g. `train/examples`
            img: batch of grayscale images.
            size: resize images.
            columns: number of columns when arranging the images.
        """
        img_resized = transforms.Resize(
            size, interpolation=transforms.InterpolationMode.NEAREST
        )(img[:, None, :, :])

        grid = utils.make_grid(img_resized, nrow=columns).cpu().numpy()
        self.logger.experiment.add_image(  # type: ignore
            path, grid, self.global_step, dataformats='CHW')

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
        self.bce_weight = bce_weight

    def _dice(
        self, y_hat: Float[Tensor, "n a r"], y_true: Float[Tensor, "n a r"]
    ) -> Float[Tensor, ""]:
        denominator = (
            torch.sum(y_hat * y_hat, dim=(1, 2))
            + torch.sum(y_true, dim=(1, 2)))
        numerator = 2 * torch.sum(y_hat * y_true, dim=(1, 2))
        return torch.mean(1.0 - numerator / denominator)

    def loss_func(
        self, y_hat: Float[Tensor, "n a r"], y_true: Float[Tensor, "n a r"]
    ) -> Float[Tensor, ""]:
        dice = self._dice(y_hat, y_true)
        bce = nn.functional.binary_cross_entropy(y_hat, y_true)
        return bce * self.bce_weight + dice * (1 - self.bce_weight)

    def log_radarhd(
        self, path: str, y_true: Float[Tensor, "n h w"],
        y_hat: Float[Tensor, "n h w"]
    ) -> None:
        """Log radarhd example images."""
        self.add_image(
            f"{path}/y_hat", torch.swapaxes(y_hat[:8], 1, 2),
            size=(256, 512), columns=4)
        self.add_image(
            f"{path}/y_true", torch.swapaxes(y_true[:8], 1, 2),
            size=(256, 512), columns=4)

    def training_step(self, batch, batch_idx):  # type: ignore
        """Standard lightning training step."""
        y_hat = self.model(batch['radar'])
        y_true = batch['lidar'].to(torch.float32)
        loss = self.loss_func(y_hat, y_true)

        self.log("loss/train", loss, prog_bar=True)
        if self.global_step % self.log_interval == 0:
            self.log_radarhd("train_example", y_true, y_hat)

        return loss

    def validation_step(self, batch, batch_idx):  # type: ignore
        """Standard lightning validation step."""
        y_hat = self.model(batch['radar'])
        y_true = batch['lidar'].to(torch.float32)
        loss = self.loss_func(y_hat, y_true)

        self.log("loss/val", loss, prog_bar=True)
        if batch_idx == 0:
            self.log_radarhd("val_example", y_true, y_hat.detach())

        return loss
