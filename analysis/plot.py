"""Plotting utilities."""

from abc import ABC, abstractmethod

import matplotlib.axes as mpl_axes
import numpy as np
from beartype.typing import Any, Optional
from jaxtyping import Num

from .result import ComparativeStats


def comparison_matrix(
    ax: mpl_axes.Axes, x: Num[np.ndarray, "N N"],
    label: Optional[Num[np.ndarray, "N N"]] = None,
    unit: str = "", sig: int = 3, **kwargs
) -> None:
    """Plot pairwise comparison matrix.

    Args:
        ax: matplotlib plot instatiated by the caller.
        x: input array. Must be a square matrix.
        label: different array to use as labels if passed. Otherwise, the cells
            are labeled with the values of `x`.
        unit: unit name; prepended to value labels.
        sig: number of significant figures to display in labels. Can take the
            special values `0` (the raw number is displayed), or `1` (is cas)
        the
            raw number is displayed.
        kwargs: arguments to forward to `imshow`, e.g. `cmap=`, `aspect=`.
    """
    N = np.arange(x.shape[0])
    x_float = np.copy(x).astype(np.float64)
    x_float[N, N] = np.nan
    ax.imshow(x_float, **kwargs)

    for (i, j), value in np.ndenumerate(x if label is None else label):
        if i != j:
            if sig > 0:
                text = f"{value:+.{sig}}{unit}"
            else:
                text = f"{value}{unit}"

            # Note that this is reversed, since coordinates are row-major
            # (in xy order) while images are column major
            ax.text(j, i, text, ha="center", va="center")

    ax.set_xticks(N)
    ax.set_yticks(N)
    ax.set_xticklabels([])
    ax.set_yticklabels([])


class PlotType(ABC):
    """Base class for a generic comparison plot."""

    desc: str = "unnamed plot type"

    @abstractmethod
    def plot(self, ax: mpl_axes.Axes, stats: ComparativeStats) -> None:
        """Draw plot."""
        pass


class Percent(PlotType):
    """Percent difference."""

    desc = "percent difference"

    def plot(self, ax: mpl_axes.Axes, stats: ComparativeStats) -> None:
        """Draw plot."""
        comparison_matrix(ax, stats.percent(), unit="%", cmap="coolwarm")


class ZScore(PlotType):
    """Z-score of the difference."""

    desc = "Z-score"

    def plot(self, ax: mpl_axes.Axes, stats: ComparativeStats) -> None:
        """Draw plot."""
        comparison_matrix(ax, stats.diff.zscore, unit=" se", cmap="coolwarm")


class Significance(PlotType):
    """Colored by significance, labels by z-score."""

    desc = "Z-score"

    def plot(self, ax: mpl_axes.Axes, stats: ComparativeStats) -> None:
        """Draw plot."""
        sig = (
            stats.significance(p=0.05, corrected=True)
            + stats.significance(p=0.05, corrected=False))
        comparison_matrix(
            ax, sig, label=stats.diff.zscore, unit=" se", cmap="coolwarm")


class ESS(PlotType):
    """Effective sample size."""

    desc = "effective sample size"

    def plot(self, ax: mpl_axes.Axes, stats: ComparativeStats) -> None:
        """Draw plot."""
        comparison_matrix(
            ax, stats.diff.ess, label=stats.diff.ess.astype(np.int64),
            cmap="viridis")
