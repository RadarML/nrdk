"""Radar training objectives."""

from abc import abstractmethod, ABC

import numpy as np
from torch import Tensor
from jaxtyping import Float, Shaped
from beartype.typing import NamedTuple, Union


#: Optionally reduced training metric
MetricValue = Union[Float[Tensor, ""], Float[Tensor, "batch"]]


class Metrics(NamedTuple):
    """Training objective values."""

    loss: Float[Tensor, ""]
    """Primary loss value, with any objective weighting applied."""

    metrics: dict[str, MetricValue]
    """Metrics to log; the name of each metric should be unique."""


class Objective(ABC):
    """Composable training objective."""

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
