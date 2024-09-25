"""Radar training objectives."""

from abc import ABC, abstractmethod

import numpy as np
import torch
from beartype.typing import Literal, NamedTuple, Optional, Union
from jaxtyping import Float, Shaped
from torch import Tensor

#: Optionally reduced training metric
MetricValue = Union[Float[Tensor, ""], Float[Tensor, "batch"]]


class Metrics(NamedTuple):
    """Training objective values.

    Attributes:
        loss: Primary loss value, with any objective weighting applied.
        metrics: Metrics to log; the name of each metric should be unique.
    """

    loss: Float[Tensor, ""]
    metrics: dict[str, MetricValue]


class Objective(ABC):
    """Composable training objective.

    NOTE: metrics should use `torch.no_grad()` to make sure gradients are not
    computed for non-loss metrics!
    """

    @abstractmethod
    def metrics(
        self, y_true: dict[str, Shaped[Tensor, "..."]],
        y_hat: dict[str, Shaped[Tensor, "..."]],
        reduce: bool = True, train: bool = True
    ) -> Metrics:
        """Get training metrics.

        Args:
            y_true: Named data channels (i.e. dataloader output).
            y_hat: Named model outputs; do not necessarily correspond 1:1 with
                keys in `y_true`.
            reduce: Whether to reduce metric outputs (i.e. during train/val)
                or return all (i.e. test, to compute time series statistics).
            train: Whether running in training mode (i.e. skip more expensive
                metrics).

        Returns:
            Objective metrics.
        """
        pass

    def visualizations(
        self, y_true: dict[str, Shaped[Tensor, "..."]],
        y_hat: dict[str, Shaped[Tensor, "..."]]
    ) -> dict[str, Shaped[np.ndarray, "H W 3"]]:
        """Generate visualizations.

        Args:
            y_true, y_hat: see :py:meth:`Objective.metrics`.

        Returns:
            A dict, where each key is the name of a visualization, and the
            value is a RGB images in HWC order.
        """
        return {}


class LPObjective:
    """Generic Lp loss, with optional missing value masking.

    Args:
        ord: loss order; only l1 (`1`) and l2 (`2`) are supported.
        mask: value to interpret as "NaN" or "not valid". If `None`, all values
            are assumed to be valid.
    """

    def __init__(
        self, ord: Literal[1, 2] = 1, mask: Optional[int | float] = None
    ) -> None:
        self.ord = ord
        self.mask = mask

    @staticmethod
    def _reduce(x: Shaped[Tensor, "n *s"]) -> Shaped[Tensor, "n"]:
        """Reduce a tensor over all but the first axis."""
        return torch.sum(x, dim=[i for i in range(1, len(x.shape) - 1)])

    def __call__(
        self, y_hat: Float[Tensor, "batch *spatial"],
        y_true: Float[Tensor, "batch *spatial"], reduce: bool = True
    ) -> MetricValue:

        if self.ord == 1:
            diff = torch.abs(y_hat - y_true)
        elif self.ord == 2:
            diff = (y_hat - y_true)**2
        else:
            raise NotImplementedError("Only L1 and L2 losses are implemented.")

        if self.mask is not None:
            mask = y_true != self.mask
            diff = torch.where(mask, diff, 0.0)
            n = self._reduce(mask)
        else:
            n = np.prod(diff.shape[1:])  # type: ignore

        res = torch.sum(self._reduce(diff) / n)
        if reduce:
            res = torch.mean(res)
        return res
