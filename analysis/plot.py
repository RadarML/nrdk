"""Plotting utilities."""

import matplotlib.axes as mpl_axes
import numpy as np
from beartype.typing import Optional, Sequence
from jaxtyping import Num

from .result import ComparativeStats


def comparison_matrix(
    ax: mpl_axes.Axes, x: Num[np.ndarray, "N N"],
    label: Optional[Num[np.ndarray, "N N"]] = None,
    unit: str = "", sig: int = 3, centered: bool = False, **kwargs
) -> None:
    """Plot pairwise comparison matrix.

    Args:
        ax: matplotlib plot instatiated by the caller.
        x: input array. Must be a square matrix.
        label: different array to use as labels if passed. Otherwise, the cells
            are labeled with the values of `x`.
        unit: unit name; prepended to value labels.
        sig: number of significant figures to display in labels. If `0`, is
            displayed without any special formatting.
        centered: whether the range of `x` is centered around 0.
        kwargs: arguments to forward to `imshow`, e.g. `cmap=`, `aspect=`.
    """
    if centered:
        vmax = np.max(np.abs(x))
        kwargs.update({"vmin": -vmax, "vmax": vmax})

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


def comparison_grid(
    axs: np.ndarray, compare: dict[str, ComparativeStats],
    method_names: Sequence[str] = []
) -> None:
    """Plot a grid of comparison plots.

    Each row plots a different comparison group (e.g. a different test split),
    while the first column plots the percent difference, and the second column
    the z-score, colored by significance.

    Args:
        axs: grid of axes.
        compare: comparison groups; the key should be the display name of each
            group, and the value should be a :py:class:`.ComparativeStats`.
            Note that if the value is in vector form, it is reduced (`.sum()`)
            before plotting.
        method_names: display name of methods corresponding to each of the
            computed statistics.
    """
    groups = len(compare)
    if len(axs.shape) != 2:
        raise ValueError("`axs` must be a 2D grid of plots.")
    if axs.shape[0] != groups or axs.shape[1] != 2:
        raise ValueError(
            f"For len(compare)={groups}, `axs` must have shape ({groups}, 2).")

    for row, (category, stats) in zip(axs, compare.items()):
        stats = stats.sum()
        pct = stats.percent()
        comparison_matrix(
            row[0], np.sign(pct) * np.sqrt(np.abs(pct)), label=pct,
            unit="%", cmap='coolwarm', centered=True)
        comparison_matrix(
            row[1],
            stats.significance(p=0.05, corrected=True, subgroups=groups)
            + stats.significance(p=0.05, corrected=False),
            label=stats.diff.zscore, unit=" se", cmap='coolwarm',
            vmin=-2.3, vmax=2.3)

    for ax, category in zip(axs[:, 0], compare):
        ax.set_ylabel(category)
        ax.set_yticklabels(method_names)
    for ax in axs[:, 1:].reshape(-1):
        ax.set_yticks([])

    for ax, metric in zip(axs[-1], ["percent difference", "z-score"]):
        ax.set_xlabel(metric)
        ax.set_xticklabels(method_names, rotation=90)
    for ax in axs[:-1].reshape(-1):
        ax.set_xticks([])
