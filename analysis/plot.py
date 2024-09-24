"""Plotting utilities."""

import matplotlib.axes as mpl_axes
import numpy as np
from beartype.typing import Optional
from jaxtyping import Num


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
