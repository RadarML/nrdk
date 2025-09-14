"""Lightning module."""

import logging
import os
import re
import threading
import warnings
from collections.abc import Iterable, Iterator, Mapping, Sequence
from functools import cache
from typing import Any, Callable, Generic, TypeVar, cast

import lightning
import numpy as np
import optree
import torch
from abstract_dataloader import spec
from abstract_dataloader.ext.objective import Objective
from jaxtyping import Shaped
from torch import Tensor, nn

from .logging import LoggerWithImages

TModel = TypeVar("TModel", bound=torch.nn.Module)
YTrueRaw = TypeVar("YTrueRaw")
YTrue = TypeVar("YTrue")
YPred = TypeVar("YPred")


class NRDKLightningModule(
    lightning.LightningModule, Generic[TModel, YTrueRaw, YTrue, YPred]
):
    """A generic lightning module for the neural radar development kit.

    - The `objective` should follow the
        [`abstract_dataloader.ext.objective`][abstract_dataloader.ext.objective]
        specifications; [`nrdk.objectives`][nrdk.objectives] provides a
        set of implementations.

    - The `optimizer` should be a pytorch optimizer with all arguments except
        the model parameter already bound:

        ```python title="train.py"
        optimizer = partial(torch.optim.AdamW, lr=1e-3, weight_decay=1e-4)
        ```

        ```yaml title="config.yaml"
        optimizer:
            _target_: torch.optim.AdamW
            _partial_: true
            lr: 1e-3
            weight_decay: 1e-4
        ```

    - The `transforms` should be an [abstract dataloader-compliant](
        https://wiselabcmu.github.io/abstract-dataloader/)
        [`Pipeline`][abstract_dataloader.spec] or
        [`Transform`][abstract_dataloader.spec]. If using a [`roverd`](
        https://radarml.github.io/red-rover/roverd/) dataset, the
        [`nrdk.roverd`][nrdk.roverd] module provides a compatible
        implementation.

    Args:
        model: pytorch model to train.
        objective: training objective.
        optimizer: pytorch optimizer to use, with all arguments except the
            model parameter already bound.
        transforms: data transform or pipeline to apply; potentially also a
            `nn.Module`; if `None`, no transforms are applied.
        compile: if `True`, compile the model with `torch.compile`. Note that
            this may cause a problem for typecheckers (so you should set
            `JAXTYPING_DISABLE=1`).
        vis_interval: log visualizations (from replica `0` only) every
            `vis_interval` steps; if `<=0`, disable altogether.
        vis_samples: maximum number of samples to visualize during training.

    Attributes:
        model: pytorch model used; note that this will cause all weights in
            `ADLLightningModule.state_dict()` to be prefixed by `model.`.
    """

    def __init__(
        self, model: TModel,
        objective: Objective[Tensor, YTrue, YPred],
        optimizer: Callable[
            [Iterable[nn.parameter.Parameter]], torch.optim.Optimizer],
        transforms:
            spec.Pipeline[Any, Any, YTrueRaw, YTrue]
            | spec.Transform[YTrueRaw, YTrue] | None = None,
        compile: bool = False,
        vis_interval: int = 0, vis_samples: int = 16
    ) -> None:
        super().__init__()

        if compile:
            jt_disable = os.environ.get("JAXTYPING_DISABLE", "0").lower()
            if jt_disable not in ("1", "true"):
                warnings.warn(
                    "torch.compile is currently incompatible with jaxtyping; "
                    "if you see type errors, set the environment variable "
                    "`JAXTYPING_DISABLE=1` to disable jaxtyping checks.")

            model = torch.compile(model)  # type: ignore

        self.model = model
        self.objective = objective
        self.optimizer = optimizer
        self.transforms = transforms
        self.vis_interval = vis_interval
        self.vis_samples = vis_samples

        self._log = logging.getLogger(self.__class__.__name__)

    @torch.compiler.disable
    def load_weights(
        self, path: str, rename: Sequence[Mapping[str, str | None]] = []
    ) -> tuple[list[str], list[str]]:
        """Load weights from an existing model for fine-tuning, resuming, etc.

        Substitutions should be specified as a list of dictionaries, each with
        a single key and value. If the value is `None`, the key is removed;
        otherwise, the key pattern is replaced with the value.

        !!! quote "Example Configuration"

            ```yaml
            path: results/example/baseline3/weights.pth
            rename:
            - decoder.occ3d.unpatch: null
            - decoder.occ3d: decoder.semseg
            ```

        !!! warning

            If the weights have a `model.` prefix (e.g., if they were saved
            directly from `NRDKLightningModule.state_dict()`), then this is
            always removed first.

        Args:
            path: path to model weights, possibly inside a `state_dict` and/or
                `model` sub-key.
            rename: substitutions to apply to the state dict. If resuming or
                loading for evaluation, this should be empty!

        Returns:
            A list of the `missing_keys` which were not present in the weights
                after filtering, so are not loaded.
            A list of the `unexpected_keys` provided in the loaded weights,
                but are not used by the model.`
        """
        weights = torch.load(path, weights_only=True)
        if "state_dict" in weights:
            weights = weights["state_dict"]
        if "model" in weights:
            weights = weights["model"]

        weights = {
            k[6:] if k.startswith("model.")
            else k: v for k, v in weights.items()}
        for pattern in rename:
            pat, sub = next(iter(pattern.items()))
            if sub is None:
                weights = {
                    k: v for k, v in weights.items() if not re.search(pat, k)}
            else:
                weights = {re.sub(pat, sub, k): v for k, v in weights.items()}

        missing, unexpected = self.model.load_state_dict(weights, strict=False)
        self._log.info(
            f"Loaded {len(weights) - len(unexpected)} weights from {path}.")
        if len(missing) > 0:
            self._log.warning(f"Not loaded: {missing}")
        if len(unexpected) > 0:
            self._log.error(f"Unexpected keys: {unexpected}")

        return missing, unexpected

    def forward(self, x: YTrue) -> YPred:
        return self.model(x)

    @torch.compiler.disable
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

    @torch.compiler.disable
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

    @torch.compiler.disable
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
    @torch.compiler.disable
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
            {f"{k}/val": v for k, v in metrics.items()}, sync_dist=True)
        self.log("loss/val", loss, sync_dist=True)

        if batch_idx == 0 and self.global_rank == 0:
            val_samples = self._get_val_samples()
            if val_samples is not None:
                samples_gpu = self.transform(cast(YTrueRaw, optree.tree_map(
                    lambda x: x.to(loss.device), val_samples)))  # type: ignore
                y_hat = self(samples_gpu)
                self.log_visualizations(samples_gpu, y_hat, split="val")

    def evaluate(
        self, dataset: torch.utils.data.DataLoader,
        metadata: Callable[
            [YTrue], Mapping[str, Shaped[Tensor, "batch"]]] | None = None,
        device: int | str | torch.device = 0,
    ) -> Iterator[tuple[
        dict[str, Shaped[np.ndarray, "batch"]],
        dict[str, Shaped[np.ndarray, "batch ..."]]
    ]]:
        """Evaluate model.

        Args:
            dataset: `DataLoader`-wrapped dataset to run.
            metadata: optional callable which takes in `y_true` and returns a
                dictionary of metadata values to return alongside the metrics.
            device: device to run on. This method does not implement
                distributed data parallelism; parallelism should be applied at
                the trace level.

        Yields:
            Metric values for each sample in the batch as returned by the
                objective, with an addditional `loss` key.
            Rendered outputs for each sample in the batch as returned by the
                objective's `.render(...)` method.
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
                if metadata is not None:
                    metrics.update(metadata(transformed))

                vis = self.objective.render(transformed, y_hat)
                yield {k: v.cpu().numpy() for k, v in metrics.items()}, vis

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizers; passthrough to the provided `Optimizer`."""
        return self.optimizer(self.parameters())
