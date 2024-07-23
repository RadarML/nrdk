"""Radar training objectives."""

import re
import torch
from torchvision.transforms import Resize, InterpolationMode
import lightning as L
import matplotlib
import numpy as np

from beartype.typing import Any, Optional
from jaxtyping import Shaped

import models
from .dataloader import RoverDataModule, RoverData
from . import metrics
from .utils import polar_to_bev


class BaseModule(L.LightningModule):
    """Composable training objective.

    All hyperparameters should be encapsulated by the constructor of the
    inheriting class. Environment-specific, non-hyperparameters should be
    passed in via methods.

    NOTE: the outer-most inheriting class must call `save_parameters()`!

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

    STOPPING_METRIC = "loss/val"
    DEFAULT_CMAP = "inferno"

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

        self.configure()

    def convert_image(
        self, img: Shaped[torch.Tensor, "batch h w"]
    ) -> Shaped[torch.Tensor, "batch h2 w2"]:
        """Transform image for display.

        Extending classes should implement this method in order to use
        `log_image_comparison`, which applies this method to each y_true/y_hat
        in the sample batch, and arranges them into a grid. If not overridden,
        this method is the identity (i.e. no image transformation for display).
        """
        return img

    def log_image_comparison(
        self, path: str, y_true: Shaped[torch.Tensor, "batch h w"],
        y_hat: Shaped[torch.Tensor, "batch h w"], cols: int = 8
    ) -> None:
        """Log a batch of images.

        Args:
            path: log path, e.g. `train/examples`
            y_true, y_hat: images to log.
            cols: number of rows of y_true/y_hat pairs; the number of samples
                must be evenly divisible by `cols`.
        """
        if y_true.shape[0] % cols != 0:
            print(y_true.shape, y_hat.shape, cols)
            raise ValueError(
                f"Samples {y_true.shape[0]} must be divisible by batch {cols}")

        y_true_cvt = list(self.convert_image(y_true))
        y_hat_cvt = list(self.convert_image(y_hat))

        rows = []
        while len(y_true_cvt) > 0:
            rows.append(torch.cat(y_true_cvt[:cols], dim=1))
            rows.append(torch.cat(y_hat_cvt[:cols], dim=1))
            y_true_cvt = y_true_cvt[cols:]
            y_hat_cvt = y_hat_cvt[cols:]
        grid = torch.cat(rows, dim=0).cpu().numpy()

        self.logger.experiment.add_image(  # type: ignore
            path, self.cmap(grid)[..., :3],
            self.global_step, dataformats='HWC')

    def get_dataset(self, path: str, debug: bool = False) -> RoverDataModule:
        """Get datamodule.

        Args:
            path: dataset root directory.
            debug: whether to run in debug mode.

        Returns:
            Corresponding `RoverDataModule`.
        """
        return RoverDataModule(**self.dataset, path=path, debug=debug)

    def configure(
        self, log_interval: int = 1, num_examples: int = 6,
        colors: Optional[str] = None, **kwargs
    ) -> None:
        """Set non-hyperparameter configurations."""
        self.log_interval = log_interval
        self.num_examples = num_examples

        if colors is None:
            colors = self.DEFAULT_CMAP
        self.cmap = matplotlib.colormaps[colors]

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
                    "interval": "step"}
            }
        else:
            return opt

    def evaluation_step(
        self, batch
    ) -> dict[str, Shaped[torch.Tensor, "batch"]]:
        """Evaluation step with mo metric aggregation.

        Args:
            batch: input batch.

        Returns:
            Dictionary of metrics for each entry in the batch (object-of-list);
            these metrics should preserve the input order, and not be
            aggregated in any way.
        """
        raise NotImplementedError()

    def evaluate(
        self, trace: RoverData, device=0
    ) -> dict[str, Shaped[np.ndarray, "N"]]:
        """Evaluate model with no metric aggregation.

        Args:
            trace: trace to run; nominally a single trace (i.e. a
                :class:`.RoverData` with `paths` of length 1).
            device: device to run on. This method does not implement
                distributed data parallelism; parallelism should be applied at
                the trace level.

        Returns:
            Dictionary with metric names as keys and raw metric results as
            values.
        """
        device = torch.device(device)
        results = []
        for batch in trace:
            batch_gpu = {
                k: torch.Tensor(v).to(device) for k, v in batch.items()}
            results.append(self.evaluation_step(batch_gpu))
        return {k: np.concatenate([x[k] for x in results]) for k in results[0]}


class RadarHD(BaseModule):
    """Radar -> lidar as bird's eye view (BEV) occupancy.

    Args:
        bce_weight: BCE loss weight; Dice loss is weighted `1 - bce_weight`.
        range_weighted: Whether to apply range weighting; see
            :class:`.CombinedDiceBCE` for details.
        kwargs: see :class:`.BaseModule`.
    """

    STOPPING_METRIC = "chamfer/val"
    DEFAULT_CMAP = "inferno"

    def __init__(
        self, bce_weight: float = 0.9, range_weighted: bool = True, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.loss = metrics.CombinedDiceBCE(
            bce_weight=bce_weight, range_weighted=range_weighted)
        self.chamfer = metrics.Chamfer()
        self.save_hyperparameters()

    def convert_image(
        self, img: Shaped[torch.Tensor, "batch h w"]
    ) -> Shaped[torch.Tensor, "batch h2 w2"]:
        """Convert image for display."""
        return polar_to_bev(img, height=512)

    def training_step(self, batch, batch_idx):  # type: ignore
        """Standard lightning training step."""
        y_hat = self.model(batch['radar'])
        y_true = batch['lidar'].to(torch.float32)
        loss = self.loss(y_hat, y_true)

        self.log(
            "loss/train", loss, on_step=True, on_epoch=True, sync_dist=True)
        if self.global_step % self.log_interval == 0:
            self.log_image_comparison(
                "sample/train", y_true[:self.num_examples],
                y_hat[:self.num_examples].detach())

        return loss

    def validation_step(self, batch, batch_idx):  # type: ignore
        """Standard lightning validation step."""
        if batch_idx == 0:
            samples = self.trainer.datamodule.val_samples  # type: ignore
            radar = torch.from_numpy(
                samples['radar']).to(batch['radar'].device)
            sample_true = torch.from_numpy(
                samples['lidar']).to(batch['lidar'].device)

            sample_hat = self.model(radar)
            self.log_image_comparison("sample/val", sample_true, sample_hat)

        y_hat = self.model(batch['radar'])
        val_loss = self.loss(y_hat, batch['lidar'].to(torch.float32))
        val_chamfer = self.chamfer(y_hat > 0.5, batch['lidar'])

        self.log("loss/val", val_loss, sync_dist=True)
        self.log("chamfer/val", val_chamfer.to("cuda"), sync_dist=True)

    def evaluation_step(
        self, batch
    ) -> dict[str, Shaped[torch.Tensor, "batch"]]:
        """Evaluate model with mo metric aggregation."""
        y_hat = self.model(batch['radar'])
        val_loss = self.loss(
            y_hat, batch['lidar'].to(torch.float32), reduce=False)
        val_chamfer = self.chamfer(
            y_hat > 0.5, batch['lidar'], reduce=False)
        return {"loss": val_loss, "chamfer": val_chamfer}


class RadarHD3D(BaseModule):
    """Radar -> lidar as depth estimation.

    Args:
        loss_order: Loss type (l1/l2).
    """

    STOPPING_METRIC = "loss/val"
    DEFAULT_CMAP = "viridis"

    def __init__(self, loss_order: int = 1, **kwargs) -> None:
        super().__init__(**kwargs)
        self.loss = metrics.LPDepth(ord=loss_order)
        self.save_hyperparameters()

    def convert_image(
        self, img: Shaped[torch.Tensor, "batch h w"]
    ) -> Shaped[torch.Tensor, "batch h2 w2"]:
        """Convert image for display."""
        return Resize(
            (512, 512 * 2), interpolation=InterpolationMode.NEAREST)(img)

    def training_step(self, batch, batch_idx):  # type: ignore
        """Standard lightning training step."""
        y_hat = self.model(batch['radar'])
        loss = self.loss(y_hat, batch['lidar'])

        self.log(
            "loss/train", loss, on_step=True, on_epoch=True, sync_dist=True)
        if self.global_step % self.log_interval == 0:
            self.log_image_comparison(
                "sample/train", batch['lidar'][:self.num_examples],
                y_hat[:self.num_examples].detach())

        return loss

    def validation_step(self, batch, batch_idx):  # type: ignore
        """Standard lightning validation step."""
        if batch_idx == 0:
            samples = self.trainer.datamodule.val_samples  # type: ignore
            radar = torch.from_numpy(
                samples['radar']).to(batch['radar'].device)
            y_true = torch.from_numpy(
                samples['lidar']).to(batch['lidar'].device)

            y_hat = self.model(radar)
            self.log_image_comparison("sample/val", y_true, y_hat)

        y_hat = self.model(batch['radar'])
        self.log("loss/val", self.loss(y_hat, batch['lidar']), sync_dist=True)

    def evaluation_step(
        self, batch
    ) -> dict[str, Shaped[torch.Tensor, "batch"]]:
        """Evaluate model with mo metric aggregation."""
        y_hat = self.model(batch['radar'])
        val_loss = self.loss(y_hat, batch['lidar'], reduce=False)
        return {"loss": val_loss}
