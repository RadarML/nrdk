"""Radar training objectives."""

import re
import torch
from torch import nn, Tensor
from torchvision import transforms, utils
import lightning as L

from beartype.typing import Any
from jaxtyping import Float, Num

import models


class BaseModel(L.LightningModule):
    """Composable training objective.
    
    Parameters
    ----------
    model: model name; should be a `nn.Module` in `models`.
    model_args: args to pass to the model.
    log_interval: how often to log images.
    optimizer: optimizer to use (from `torch.optim`).
    default_lr: default learning rate to use.
    optimizer_args: per-layer/parameter optimizer arguments; each key is a
        regex, and each value are additional parameters to pass to matching
        parameters.
    """

    def __init__(
        self, model: str, model_args: dict[str, Any], log_interval: int = 500,
        optimizer: str = "AdamW", default_lr: float = 1e-3,
        optimizer_args: dict[str, dict] = {}
    ) -> None:
        super().__init__()

        self.model = getattr(models, model)(**model_args)

        self.save_hyperparameters()
        self.log_interval = log_interval
        self.optimizer = optimizer
        self.optimizer_args = optimizer_args
        self.default_lr = default_lr

    def add_image(
        self, path: str, img: Num[torch.Tensor, "n h w"],
        size: tuple[int, int] = (256, 1024), columns: int = 8
    ) -> None:
        """Log a batch of images.
        
        Parameters
        ----------
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

        return getattr(torch.optim, self.optimizer)([
            {"params": subsets[k], **self.optimizer_args[k]}
            for k in subsets
        ] + [{"params": nomatch}], lr=self.default_lr)


class RadarHD(BaseModel):
    """Radar -> lidar.
    
    Parameters
    ----------
    bce_weight: BCE loss weight for positive cells.
    kwargs: see `BaseModel`.
    """

    def __init__(self, bce_weight: float = 0.9, **kwargs) -> None:
        super().__init__(**kwargs)
        self.bce_weight = bce_weight

    def loss_func(
        self, y_hat: Float[Tensor, "n ..."], y_true: Float[Tensor, "n ..."]
    ) -> Float[Tensor, ""]:
        weight = self.bce_weight * y_true + (1 - self.bce_weight)
        return nn.functional.binary_cross_entropy(y_hat, y_true, weight=weight)

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

    def training_step(self, batch, batch_idx):
        y_hat = self.model(batch['radar'])
        y_true = batch['lidar'].to(torch.float32)
        loss = self.loss_func(y_hat, y_true)

        self.log("loss/train", loss, prog_bar=True)
        if self.global_step % self.log_interval == 0:
            self.log_radarhd("train_example", y_true, y_hat)

        return loss

    def validation_step(self, batch, batch_idx):
        y_hat = self.model(batch['radar'])
        y_true = batch['lidar'].to(torch.float32)
        loss = self.loss_func(y_hat, y_true)

        self.log("loss/val", loss, prog_bar=True)
        if batch_idx == 0:
            self.log_radarhd("val_example", y_true, y_hat.detach())

        return loss
