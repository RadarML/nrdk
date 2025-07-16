"""Pytorch lightning callbacks."""

from time import perf_counter

import torch
from lightning.pytorch.callbacks import Callback


class PerformanceMonitor(Callback):
    """Callback to log performance statistics.

    Logs the following metrics (namespaced with the specified `name`):

    - `step_time`: duration of each step, in seconds
    - `throughput`: training step throughput, in steps per second.
    - `{allocated|reserved|active|inactive_split}.{global_rank}`: memory usage
        statistics for each GPU, in the specified `units` (default: GB).

    Args:
        units: memory units to use for logging; defaults to 1e9 (GB).
        name: namespace to log metric under (i.e., `{name}/{metric}`).
    """

    def __init__(self, units: float = 1e9, name: str = "perf") -> None:
        self.units = units
        self.name = name

    def on_train_batch_start(
        self, trainer, pl_module, batch, batch_idx
    ) -> None:
        self._start = perf_counter()

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx
    ) -> None:
        duration = perf_counter() - self._start
        pl_module.log(f"{self.name}/step_time", duration, on_step=True)
        pl_module.log(f"{self.name}/throughput", 1 / duration, on_step=True)

        stats = torch.cuda.memory_stats(device=trainer.global_rank)
        for metric in ["allocated", "reserved", "active", "inactive_split"]:
            pl_module.log(
                f"{self.name}/{metric}.{trainer.global_rank}",
                stats[f"{metric}_bytes.all.current"] / self.units,
                on_step=True)
