"""Lightning module."""

import threading
import warnings
from collections.abc import Iterator
from functools import cache
from typing import Any, Generic, TypeVar, cast

import lightning
import numpy as np
import optree
import torch
from abstract_dataloader import spec
from abstract_dataloader.ext.objective import Objective
from jaxtyping import Shaped
from torch import Tensor

from .logging import LoggerWithImages

TModel = TypeVar("TModel", bound=torch.nn.Module)
YTrueRaw = TypeVar("YTrueRaw")
YTrue = TypeVar("YTrue")
YPred = TypeVar("YPred")


class ADLLightningModule(
    lightning.LightningModule, Generic[TModel, YTrueRaw, YTrue, YPred]
):
    """A generic lightning module using ADL objectives.

    Args:
        model: pytorch model to train.
        objective: training objective.
        transforms: data transform or pipeline to apply; potentially also a
            `nn.Module`; if `None`, no transforms are applied.
        vis_interval: log visualizations (from replica `0` only) every
            `vis_interval` steps; if `<=0`, disable altogether.
        vis_samples: maximum number of samples to visualize during training.
    """

    def __init__(
        self, model: TModel,
        objective: Objective[Tensor, YTrue, YPred],
        transforms:
            spec.Pipeline[Any, Any, YTrueRaw, YTrue]
            | spec.Transform[YTrueRaw, YTrue] | None = None,
        vis_interval: int = 0, vis_samples: int = 16
    ) -> None:
        super().__init__()
        self.model = model
        self.objective = objective
        self.transforms = transforms
        self.vis_interval = vis_interval
        self.vis_samples = vis_samples

    def forward(self, x: YTrue) -> YPred:
        return self.model(x)

    def _make_log(
        self, y_true: YTrue, y_pred: YPred, split: str, step: int
    ) -> None:
        """Inner thread callable for `log_visualizations`.

        Pseudo-closure to avoid closure-induced memory leaks in pytorch.
        """
        with torch.no_grad():
            images = self.objective.visualizations(y_true, y_pred)
        images = {f"{k}/{split}": v for k, v in images.items()}

        if len(images) > 0:
            if not isinstance(self.logger, LoggerWithImages):
                warnings.warn(
                    "Tried to log visualizations, but the logger does not "
                    "implement the `LoggerWithImages` interface.")
            else:
                 self.logger.log_images(images, step=step)

    def log_visualizations(
        self, y_true: YTrue, y_pred: YPred, split: str = "train"
    ) -> None:
        """Log all image visualizations.

        This method transfers data to CPU memory (up to `vis_samples`), and
        spins up a thread to asynchronously generate `.visualizations(...)` for
        each objective.

        !!! warning

            During multi-gpu training, the caller is responsible for only
            calling `log_visualizations` on one worker:

            ```python
            if self.global_rank == 0:
                self.log_visualizations(...)
            ```

        Args:
            y_true: ground truth input values.
            y_pred: model output values.
            split: train/val split to put in the output path.
        """
        y_true, y_pred = optree.tree_map(
            lambda x: x[:self.vis_samples].cpu().detach(),
            (y_true, y_pred))  # type: ignore
        threading.Thread(
            target=self._make_log,
            args=(y_true, y_pred, split, self.global_step)).start()

    def transform(self, batch: YTrueRaw) -> YTrue:
        """Apply transforms."""
        with torch.no_grad():
            if isinstance(self.transforms, spec.Pipeline):
                return self.transforms.batch(batch)
            elif isinstance(self.transforms, spec.Transform):
                return self.transforms(batch)
            else:  # YTrue = YTrueRaw
                return cast(YTrue, batch)

    def training_step(
        self, batch: YTrueRaw, batch_idx: int
    ) -> torch.Tensor:
        """Standard lightning training step."""
        transformed = self.transform(batch)
        y_pred = self(transformed)

        loss, metrics = self.objective(transformed, y_pred, train=True)
        loss = torch.mean(loss)
        metrics = {k: torch.mean(v) for k, v in metrics.items()}

        self.log_dict(
            {f"{k}/train": v for k, v in metrics.items()},
            on_step=True, on_epoch=True, sync_dist=True)
        self.log(
            "loss/train", loss,
            on_step=True, on_epoch=True, sync_dist=True)

        do_log = (
            self.global_rank == 0
            and self.vis_interval > 0
            and (self.global_step % self.vis_interval == 0))
        if do_log:
            self.log_visualizations(transformed, y_pred, split="train")

        return loss

    @cache
    def _get_val_samples(self) -> YTrueRaw | None:
        """Get validation samples, and fail gracefully."""
        try:
            datamodule = getattr(self.trainer, "datamodule")
            return getattr(datamodule, "samples", None)
        # fallback case where even trainer.datamodule does not exist
        except AttributeError:
            return None

    def validation_step(self, batch: YTrueRaw, batch_idx: int) -> None:
        """Standard lightning validation step."""
        transformed = self.transform(batch)
        y_hat = self(transformed)
        loss, metrics = self.objective(transformed, y_hat, train=False)
        loss = torch.mean(loss)
        metrics = {k: torch.mean(v) for k, v in metrics.items()}

        self.log_dict(
            {f"{k}/train": v for k, v in metrics.items()}, sync_dist=True)
        self.log("loss/train", loss, sync_dist=True)

        if batch_idx == 0 and self.global_rank == 0:
            val_samples = self._get_val_samples()
            if val_samples is not None:
                samples_gpu = self.transform(cast(YTrueRaw, optree.tree_map(
                    lambda x: x.to(loss.device), val_samples)))  # type: ignore
                y_hat = self(samples_gpu)
                self.log_visualizations(samples_gpu, y_hat, split="val")

    def evaluate(
        self, dataset: torch.utils.data.DataLoader,
        device: int | str | torch.device = 0,
    ) -> Iterator[tuple[
        dict[str, Shaped[np.ndarray, "batch"]],
        dict[str, Shaped[np.ndarray, "batch ..."]]
    ]]:
        """Evaluate model.

        Args:
            dataset: `DataLoader`-wrapped dataset to run.
            device: device to run on. This method does not implement
                distributed data parallelism; parallelism should be applied at
                the trace level.

        Yields:
            Metric values for each sample in the batch as returned by the
                objective, with an addditional `loss` key.
            Rendered outputs for each sample in the batch as returned by the
                objective's `.visualizations(...)` method.
        """
        # Don't forget this step!
        self.eval()

        device = torch.device(device)
        with torch.no_grad():
            for batch in dataset:
                batch_gpu = optree.tree_map(lambda x: x.to(device), batch)

                transformed = self.transform(cast(YTrueRaw, batch_gpu))
                y_hat = self(transformed)

                loss, metrics = self.objective(transformed, y_hat, train=False)
                metrics["loss"] = loss

                vis = self.objective.visualizations(transformed, y_hat)
                yield {k: v.cpu().numpy() for k, v in metrics.items()}, vis
