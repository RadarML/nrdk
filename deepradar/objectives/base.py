"""Radar training objectives."""

from abc import ABC, abstractmethod

import numpy as np
import torch
from beartype.typing import Any, Literal, NamedTuple, Optional, Union, cast
from jaxtyping import Bool, Float, Shaped
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
            value is a single RGB images in HWC order.
        """
        return {}

    RENDER_CHANNELS: dict[str, dict[str, Any]] = {}
    """Channels output by :py:meth:`Objective.render`.

    Each key should correspond to a channel name, and each value should
    correspond to a configuration for :py:mod:`roverd.channels`.
    """

    def render(
        self, y_true: dict[str, Shaped[Tensor, "batch ..."]],
        y_hat: dict[str, Shaped[Tensor, "batch ..."]]
    ) -> dict[str, Shaped[np.ndarray, "batch ..."]]:
        """Summarize predictions to visualize later.

        Args:
            y_true, y_hat: see :py:meth:`Objective.metrics`.

        Returns:
            A dict, where each key is the name of a visualization or output
            data, and the value is a quantized or packed format if possible.
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
        return torch.sum(x.reshape(x.shape[0], -1), dim=1)

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

        res = self._reduce(diff) / n
        return torch.mean(res) if reduce else res


class PointCloudObjective(ABC):
    """Generic point-cloud objective (e.g. Chamfer, Hausdorff).

    Note that this base class only supplies the point cloud framework;
    inheritors must implement a `as_points` method, which specifies the
    conversion from the raw inputs to cartesian points.

    Supported modes:

    - `chamfer` (default): chamfer distance (mean)
    - `hausdorff`: hausdorff distance (max)
    - `modhausdorff`: modified hausdorff distance (median)

    Args:
        mode: specified modes.
        on_empty: value to use if one of the provided maps is completely empty.
    """

    def __init__(self, on_empty: float = 64.0, mode: str = "chamfer") -> None:
        self.mode = mode
        self.on_empty = on_empty

    @abstractmethod
    def as_points(self, data: Shaped[Tensor, "..."]) -> Float[Tensor, "n d"]:
        """Convert a model's native representation to cartesian points.

        Note that since the output number of points may vary across a batch,
        this method is not batched.
        """

    @staticmethod
    def distance(
        x: Float[Tensor, "n1 d"], y: Float[Tensor, "n2 d"]
    ) -> Float[Tensor, "n1 n2"]:
        """Compute the pairwise N-dimension l2 distances between x and y."""
        return torch.sqrt(torch.sum(torch.square(
            x[:, None, :] - y[None, :, :]
        ), dim=-1))

    def _forward(self, x, y):
        pts_x = self.as_points(x)
        pts_y = self.as_points(y)
        dist = self.distance(pts_x, pts_y)

        if dist.shape[0] == 0 or dist.shape[1] == 0:
            return torch.full((), self.on_empty, device=x.device)

        d1 = torch.min(dist, dim=0).values
        d2 = torch.min(dist, dim=1).values

        if self.mode == "modhausdorff":
            return torch.median(torch.concatenate([d1, d2]))
        elif self.mode == "chamfer":
            return (torch.mean(d1) + torch.mean(d2)) / 2
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    def __call__(
        self, y_hat: Shaped[Tensor, "..."], y_true: Shaped[Tensor, "..."],
        reduce: bool = True
    ) -> MetricValue:
        """Compute chamfer distance, in range bins."""
        _iter = (self._forward(xs, ys) for xs, ys in zip(y_hat, y_true))
        if reduce:
            return cast(Float[Tensor, ""], sum(_iter)) / y_hat.shape[0]
        else:
            return torch.stack(list(_iter))


def accuracy_metrics(
    y_hat: Bool[Tensor, "batch ..."], y_true: Bool[Tensor, "batch ..."],
    prefix: str = "", reduce: bool = True
) -> dict[str, Float[Tensor, "*#batch"]]:
    """Generic accuracy-related metrics."""
    batch = y_hat.shape[0]
    y_hat = y_hat.reshape(batch, -1)
    y_true = y_true.reshape(batch, -1)
    n = y_hat.shape[1]

    tpr = torch.sum(y_hat & y_true, dim=1) / n
    tnr = torch.sum(~y_hat & ~y_true, dim=1) / n
    fpr = torch.sum(y_hat & ~y_true, dim=1) / n
    fnr = torch.sum(~y_hat & y_true, dim=1) / n

    metrics = {
        "tpr": tpr, "tnr": tnr, "fpr": fpr, "fnr": fnr,
        "acc": tpr + tnr,
        "precision": tpr / (tpr + fpr),
        "recall": tpr / (tpr + fnr),
        "f1": 2 * tpr / (2 * tpr + fpr + fnr),
    }

    if reduce:
        metrics = {k: torch.mean(v) for k, v in metrics.items()}

    return {prefix + k: v for k, v in metrics.items()}
