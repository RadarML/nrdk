"""Pytorch lightning callbacks."""

from time import perf_counter

import numpy as np
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


class GradientStats(Callback):
    """Callback to log gradient statistics.

    Logs the following metrics with respect to the l2 norm of gradients
    (namespaced with the specified `name`), calculated across the specified
    `interval`:
    - `mean`: mean across the specified interval
    - `std`: standard deviation
    - `min`, `max`: minimum and maximum values

    Args:
        interval: interval (in steps) over which to calculate statistics.
        name: namespace to log metric under (i.e., `{name}/{metric}`).
    """

    def __init__(self, interval: int = 100, name: str = "grad_norm") -> None:
        self.interval = interval
        self.name = name

        self.__reset_stats()

    def __reset_stats(self) -> None:
        self.m1 = 0.0
        self.m2 = 0.0
        self.min = float('inf')
        self.max = 0.0

    def on_after_backward(self, trainer, pl_module) -> None:
        # Note that gradients should be synchronized across shards already!
        total_norm = torch.nn.utils.get_total_norm(
            p.grad for p in pl_module.parameters() if p.grad is not None)

        norm = torch.tensor(total_norm).item()
        self.m1 += norm
        self.m2 += norm**2
        self.min = min(self.min, norm)
        self.max = max(self.max, norm)

        if (trainer.global_step + 1) % self.interval == 0:
            mean = self.m1 / self.interval

            stats = {
                f"{self.name}/mean": mean,
                f"{self.name}/std": np.sqrt(
                    (self.m2 / self.interval) - mean**2),
                f"{self.name}/min": self.min,
                f"{self.name}/max": self.max,
            }
            # Synchronization not necessary!
            pl_module.log_dict(stats, on_step=True, sync_dist=False)

            self.__reset_stats()
