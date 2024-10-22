"""Training objective."""

import json
import os
import threading
from queue import Queue

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
        decoder: decoder specifications; note that the `dim` input is fetched
            from the encoder specification.
        optimizer: optimizer specifications; see :py:func:`.create_optimizer`.
    """

    def __init__(
        self, dataset: dict = {}, objectives: list[dict] = [],
        encoder: dict = {}, decoder: dict = {}, optimizer: dict = {}
    ) -> None:
        super().__init__()

        self.dataset = dataset
        self.objectives: list[mod_objectives.Objective] = [
            getattr(mod_objectives, spec["name"])(**spec["args"])
            for spec in objectives]
        self.encoder = getattr(models, encoder["name"])(**encoder["args"])
        self.decoder = getattr(models, decoder["name"])(
            dim=encoder["args"]["dim"], **decoder["args"])
        self.optimizer = optimizer

        self.configure()
        self.save_hyperparameters()

        # warn_tensor_cycles()

    @classmethod
    def load_from_experiment(
        cls, path: str, checkpoint: Optional[str] = None
    ) -> "DeepRadar":
        """Load from experiment directory.

        - If `checkpoint` is specified, that checkpoint is loaded.
        - Otherwise, if a `meta.json` is available, the `best` checkpoint
          in `meta.json` is loaded.
        - Otherwise, `last.ckpt` is loaded.

        Args:
            path: experiment directory; should contain a `hparams.yaml` file
                and `checkpoints` directory.
            checkpoint: target checkpoint name (file in `<path>/checkpoints`).
        """
        if checkpoint is None:
            try:
                with open(os.path.join(path, "meta.json")) as f:
                    checkpoint = str(json.load(f)["best"])
                print(f"Using best checkpoint: {checkpoint}")
            except FileNotFoundError:
                print("No `meta.json` file found. Defaulting to `last.ckpt`.")
                checkpoint = "last.ckpt"
        else:
            print(f"Using specified checkpoint: {checkpoint}")

        return cls.load_from_checkpoint(
            os.path.join(path, "checkpoints", checkpoint),
            hparams_file=os.path.join(path, "hparams.yaml"))

    def get_dataset(
        self, path: str, n_workers: Optional[int] = None
    ) -> RoverDataModule:
        """Get datamodule.

        Args:
            path: dataset root directory.
            n_workers: number of workers. Use `0` to debug; leave `None` to
                use the number of CPUs.

        Returns:
            Corresponding `RoverDataModule`.
        """
        return RoverDataModule(**self.dataset, path=path, n_workers=n_workers)

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
        return self.decoder(encoded)

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
                self.log(
                    f"{k}/train", v,
                    on_step=True, on_epoch=True, sync_dist=True)
        self.log(
            "loss/train", loss, on_step=True, on_epoch=True, sync_dist=True)

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
        self, batch, render: bool = False
    ) -> tuple[
        dict[str, Shaped[np.ndarray, "batch"]],
        dict[str, Shaped[np.ndarray, "batch ..."]]
    ]:
        """Evaluation step with mo metric aggregation.

        Args:
            batch: input batch.
            render: whether to perform rendering as well instead of only
                computing metrics.

        Returns:
            A tuple with a dictionary of metrics and a dictionary of
            rendered values for each entry in the batch (object-of-list). No
            mean/sum aggregation is performed, and the input order is
            preserved. If `render=False`, the rendered values dict is empty.
        """
        y_hat = self.forward(batch)

        loss = 0.0
        metrics = {}
        rendered = {}
        for objective in self.objectives:
            m = objective.metrics(batch, y_hat, train=False, reduce=False)
            loss += m.loss.cpu().numpy()  # type: ignore
            metrics.update({k: v.cpu().numpy() for k, v in m.metrics.items()})

            if render:
                rendered.update(objective.render(batch, y_hat))

        metrics["loss"] = loss  # type: ignore
        return metrics, rendered

    def evaluate(
        self, trace: DataLoader, device=0, desc: Optional[str] = None,
        outputs: Optional[
            dict[str, Queue[Optional[Shaped[np.ndarray, "..."]]]]] = None
    ) -> dict[str, Shaped[np.ndarray, "N"]]:
        """Evaluate model with no metric aggregation.

        Args:
            trace: `DataLoader`-wrapped trace to run; nominally a single trace
                (i.e. a :class:`.RoverData` with `paths` of length 1).
            device: device to run on. This method does not implement
                distributed data parallelism; parallelism should be applied at
                the trace level.
            desc: progress bar description (display only).
            outputs: output queues for rendering. If not specified, no
                rendering is performed -- only evaluation. Should contain a
                dictionary of queues; each rendered value is placed
                in the queue inside `outputs` with the same key. Another
                caller-managed thread is then responsible for handling the
                queues (e.g. :py:meth:`roverd.Channel.consume`). `None` values
                are put in the queues to indicate termination.

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

                metrics, rendered = self.evaluation_step(
                    batch_gpu, render=(outputs is not None))
                results.append(metrics)
                if outputs is not None:
                    for k, v in rendered.items():
                        for sample in v:
                            outputs[k].put(sample)

        if outputs is not None:
            for q in outputs.values():
                q.put(None)

        return {k: np.concatenate([x[k] for x in results]) for k in results[0]}
