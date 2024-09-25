"""Training objective."""

import threading

import lightning as L
import numpy as np
import torch
from beartype.typing import Optional
from jaxtyping import Shaped
from torch.utils.data import DataLoader
from tqdm import tqdm

import models
from deepradar import objectives as mod_objectives
from deepradar.dataloader import RoverDataModule
from deepradar.optimizer import create_optimizer

# from torch.utils.viz._cycles import warn_tensor_cycles


class DeepRadar(L.LightningModule):
    """Composable training objective.

    All hyperparameters should be encapsulated by the constructor of the
    inheriting class. Environment-specific, non-hyperparameters should be
    passed in via methods.

    Args:
        dataset: dataset specifications; see :py:class:`.RoverDataModule`.
        objectives: list of objective specifications; see
            :py:mod:`deepradar.objectives`.
        encoder: encoder specification; see :py:mod:`models`.
        decoders: list of decoder specifications; see :py:mod:`models`; the
            `dim` (embedding dimension) specification is fetched from the
            encoder specification, and should not be included in `decoders`.
        optimizer: optimizer specifications; see :py:func:`.create_optimizer`.
    """

    def __init__(
        self, dataset: dict = {}, objectives: list[dict] = [],
        encoder: dict = {}, decoders: list[dict] = [],
        optimizer: dict = {}
    ) -> None:
        super().__init__()

        self.dataset = dataset
        self.objectives: list[mod_objectives.Objective] = [
            getattr(mod_objectives, spec["name"])(**spec["args"])
            for spec in objectives]
        self.encoder = getattr(models, encoder["name"])(**encoder["args"])
        self.decoders = torch.nn.ModuleList(
            getattr(
                models, spec["name"]
            )(**spec["args"], dim=encoder["args"]["dim"])
            for spec in decoders)
        self.optimizer = optimizer

        self.configure()
        self.save_hyperparameters()

        # warn_tensor_cycles()

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
    ) -> None:
        """Set non-hyperparameter configurations."""
        self.log_interval = log_interval
        self.num_examples = num_examples

    def configure_optimizers(self):  # type: ignore
        """Configure optimizers."""
        return create_optimizer(self, **self.optimizer)

    def forward(self, batch):
        """Apply model."""
        encoded = self.encoder(batch['radar'])
        y_hat = {}
        for decoder in self.decoders:
            y_hat.update(decoder(encoded))
        return y_hat

    def _make_log(self, y_true, y_hat, split, step: int) -> None:
        """Inner thread callable for `log_visualizations`.

        Pseudo-closure to avoid closure-induced memory leaks in pytorch.
        """
        for objective in self.objectives:
            with torch.no_grad():
                imgs = objective.visualizations(y_true, y_hat)
            for k, v in imgs.items():
                self.logger.experiment.add_image(  # type: ignore
                    f"{k}/{split}", v, step, dataformats='HWC')

    def log_visualizations(self, y_true, y_hat, split: str = 'train') -> None:
        """Log all image visualizations.

        This method transfers data to CPU memory (up to `num_examples`), and
        spins up a thread to asynchronously generate `.visualizations(...)` for
        each objective.

        **NOTE**: during multi-gpu training, the caller is responsible for only
        calling on one worker::

            if self.global_rank == 0:
                self.log_visualizations(...)

        Args:
            y_true, y_hat: input, output values.
            split: train/val split to put in the output path.
        """
        y_true = {
            k: v[:self.num_examples].cpu().detach()
            for k, v in y_true.items()}
        y_hat = {
            k: v[:self.num_examples].cpu().detach()
            for k, v in y_hat.items()}
        threading.Thread(
            target=self._make_log,
            args=(y_true, y_hat, split, self.global_step)).start()

    def log_debug_stats(self, unit: float = 1024 * 1024 * 1024) -> None:
        """Log memory statistics for each GPU.

        Args:
            unit: Memory unit; defaults to GiB (`/2**30`)
        """
        stats = torch.cuda.memory_stats(device=self.global_rank)
        for metric in ["allocated", "reserved", "active", "inactive_split"]:
            for pool in ["all", "large_pool", "small_pool"]:
                self.log(
                    f"debug/{metric}.{pool}.{self.global_rank}",
                    stats[f"{metric}_bytes.{pool}.current"] / unit,
                    on_step=True)

    def training_step(self, batch, batch_idx):  # type: ignore
        """Standard lightning training step."""
        y_hat = self.forward(batch)

        loss = 0.0
        for objective in self.objectives:
            metrics = objective.metrics(batch, y_hat, train=True, reduce=True)
            loss += metrics.loss
            for k, v in metrics.metrics.items():
                self.log(f"{k}/train", v, on_step=True, sync_dist=True)
        self.log("loss/train", loss, on_step=True, sync_dist=True)

        if self.global_rank == 0:
            if self.global_step % self.log_interval == 0:
                self.log_visualizations(batch, y_hat, split="train")
        self.log_debug_stats()

        return loss

    def validation_step(self, batch, batch_idx):  # type: ignore
        """Standard lightning validation step."""
        y_hat = self.forward(batch)

        loss = 0.0
        for objective in self.objectives:
            metrics = objective.metrics(batch, y_hat, train=False, reduce=True)
            loss += metrics.loss
            for k, v in metrics.metrics.items():
                self.log(f"{k}/val", v, sync_dist=True)
        self.log("loss/val", loss, sync_dist=True)

        if batch_idx == 0 and self.global_rank == 0:
            samples = self.trainer.datamodule.val_samples  # type: ignore
            samples_torch = {
                k: torch.from_numpy(v).to(batch['radar'].device)
                for k, v in samples.items()}
            y_hat = self.forward(samples_torch)
            self.log_visualizations(samples_torch, y_hat, split="val")

    def evaluation_step(
        self, batch
    ) -> dict[str, Shaped[torch.Tensor, "batch"]]:
        """Evaluation step with mo metric aggregation.

        Args:
            batch: input batch.

        Returns:
            Dictionary of metrics for each entry in the batch (object-of-list)
            without aggregation, with the input order preserved.
        """
        y_hat = self.forward(batch)

        loss = 0.0
        metrics = {}
        for objective in self.objectives:
            m = objective.metrics(batch, y_hat, train=False, reduce=False)
            loss += m.loss  # type: ignore
            metrics.update(m.metrics)
        metrics["loss"] = loss  # type: ignore
        return metrics

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
