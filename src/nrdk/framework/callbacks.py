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

    !!! info

        This is equivalent to
        [`DeviceStatsMonitor`][lightning.pytorch.callbacks.], though with a
        bit more polish and only recording at the end of each batch.

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
        if trainer.global_rank == 0:
            self._start = perf_counter()

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx
    ) -> None:
        if trainer.global_rank == 0:
            duration = perf_counter() - self._start
            pl_module.log_dict({
                f"{self.name}/step_time": torch.tensor(duration),
                f"{self.name}/throughput": torch.tensor(1 / duration)
            }, on_step=True)

        stats = torch.cuda.memory_stats(device=trainer.global_rank)
        stats = {
            f"{self.name}/{k}.{trainer.global_rank}":
            stats[f"{k}_bytes.all.current"] / self.units
            for k in ["allocated", "reserved", "active", "inactive_split"]
        }
        pl_module.log_dict(stats, on_step=True)
