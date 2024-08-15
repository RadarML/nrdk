"""Radar training objectives."""

import re
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import lightning as L
import matplotlib
import numpy as np

from beartype.typing import Any, Optional, Union
from jaxtyping import Shaped, Float

import models
from deepradar.dataloader import RoverDataModule


#: Optionally reduced training metric
MetricValue = Union[Float[torch.Tensor, ""], Float[torch.Tensor, "batch"]]


class Module(L.LightningModule):
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
        self, trace: DataLoader, device=0, desc: Optional[str] = None
    ) -> dict[str, Shaped[np.ndarray, "N"]]:
        """Evaluate model with no metric aggregation.

        Args:
            trace: `DataLoader`-wrapped trace to run; nominally a single trace
                (i.e. a :class:`.RoverData` with `paths` of length 1).
            device: device to run on. This method does not implement
                distributed data parallelism; parallelism should be applied at
                the trace level.
            desc: progress bar description (display only).

        Returns:
            Dictionary with metric names as keys and raw metric results as
            values.
        """
        device = torch.device(device)
        results = []
        for batch in tqdm(trace, desc=desc):
            with torch.no_grad():
                batch_gpu = {
                    k: torch.Tensor(v).to(device) for k, v in batch.items()}
                results.append(self.evaluation_step(batch_gpu))
        return {
            k: np.concatenate([x[k].cpu().numpy() for x in results])
            for k in results[0]}
