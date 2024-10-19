"""Plotting utilities."""

import matplotlib.axes as mpl_axes
import numpy as np
from beartype.typing import Optional, Sequence
from jaxtyping import Num
from scipy.stats import norm

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
    method_names: Sequence[str] = [], shortnames: bool = False,
    cmap: str = "coolwarm", **kwargs
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
    if axs.shape[1] != groups or axs.shape[0] != 3:
        raise ValueError(
            f"For len(compare)={groups}, `axs` must have shape ({groups}, 3).")

    for row, (category, stats) in zip(axs.T, compare.items()):
        stats = stats.sum()
        pct = stats.percent()
        comparison_matrix(
            row[1], np.sign(pct) * np.sqrt(np.abs(pct)), label=pct,
            unit="%", cmap=cmap, centered=True, **kwargs)
        comparison_matrix(
            row[2],
            stats.significance(p=0.05, corrected=True, subgroups=groups)
            + stats.significance(p=0.05, corrected=False),
            label=stats.diff.zscore, unit=" se", cmap=cmap,
            vmin=-2.3, vmax=2.3, **kwargs)


    lower = None
    upper = None
    for ax, stats in zip(axs[0], compare.values()):
        stats = stats.sum()
        y = stats.abs.mean
        yerr = norm.ppf(1 - 0.05 / 2) * stats.abs.stderr
        ax.errorbar(
            np.arange(len(method_names)), y, yerr=yerr, capsize=10)
        ax.set_xticks(np.arange(len(method_names)))
        ax.grid()
        ax.set_xlim(-0.5, len(method_names) - 0.5)

        if lower is None:
            lower = min(y - yerr)
            upper = max(y + yerr)
        else:
            lower = min(lower, min(y - yerr))
            upper = max(upper, max(y + yerr))

    axs[0, 0].set_ylabel("mean")
    if upper is not None and lower is not None:
        margin = 0.05 * (upper - lower)
        lower -= margin
        upper += margin
    for ax in axs[0]:
        ax.set_ylim(lower, upper)
        for tick in ax.xaxis.get_major_ticks():
            tick.tick1line.set_visible(False)
            tick.tick2line.set_visible(False)
            tick.label1.set_visible(False)
            tick.label2.set_visible(False)
    for ax in axs[0, 1:]:
        for tick in ax.yaxis.get_major_ticks():
            tick.tick1line.set_visible(False)
            tick.tick2line.set_visible(False)
            tick.label1.set_visible(False)
            tick.label2.set_visible(False)

    for ax, category in zip(axs[-1], compare):
        ax.set_xlabel(category)
        ax.set_xticklabels(method_names, rotation=None if shortnames else 90)
    for ax in axs[1:-1].reshape(-1):
        ax.set_xticks([])

    for ax, metric in zip(axs[1:, 0], ["difference", "z-score"]):
        ax.set_ylabel(metric)
        ax.set_yticklabels(method_names)
    for ax in axs[1:, 1:].reshape(-1):
        ax.set_yticks([])
