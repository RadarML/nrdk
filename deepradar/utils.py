"""Miscellaneous utilities."""

import matplotlib
import numpy as np
import torch

from torch import Tensor
from jaxtyping import Shaped, Num


def polar_to_bev(
    data: Shaped[Tensor, "batch azimuth range"], height: int = 512
) -> Shaped[Tensor, "batch height width"]:
    """Convert polar image to a birds-eye-view cartesian image.

    Uses nearest-neighbor interpolation for a specified resolution.

    Assumptions:

    - The provided `data` represents front-facing angles only, with bin 0 on
      the left moving clockwise to bin -1 on the right.
    - The provided range starts from 0, and increases.
    - The specified `height` is much greater (e.g. 2x) than the max range.
      Otherwise, interpolation artifacts can appear.

    Args:
        data: batched polar grid in batch-azimuth-range order. Can be any
            data type (e.g. `float` intensity image or `bool` occupancy grid).
        height: desired height of the BEV image.

    Returns:
        Nearest-neighbor-interpolated BEV image, with resolution
        `(height, height * 2)`. The data type is always preserved.
    """
    rmax = data.shape[2] - 1
    thetamax = data.shape[1] - 1

    x, y = torch.meshgrid(
        torch.linspace(-rmax, rmax, height * 2, device=data.device),
        torch.linspace(0, rmax, height, device=data.device), indexing='xy')
    r = torch.sqrt(x**2 + y**2)
    theta = torch.acos(x / r)

    ir = torch.floor(r).to(dtype=torch.int)
    itheta = torch.floor(theta / torch.pi * thetamax).to(dtype=torch.int)

    # Handle out-of-bounds values in the corners of the image
    mask = (ir > rmax)
    ir[mask] = 0
    bev = data[:, itheta, ir]
    bev[:, mask] = 0

    return bev


def comparison_grid(
    y_true: Shaped[Tensor, "batch h w"], y_hat: Shaped[Tensor, "batch h w"],
    cols: int = 8, cmap: str = 'viridis'
) -> Num[np.ndarray, "h2 w2 3"]:
    """Create image comparison grid.
    
    Args:
        y_true, y_hat: images to compare; nominally y_true/y_hat, though the
            exact order does not matter.
        cols: number of columns. Any excess images (which do not fill a full
            row) will be discarded.
        cmap: matplotlib colormap to use, e.g. `viridis`, `inferno`. Note that
            the alpha channel is discarded.
    
    Returns:
        Colormapped grid of the inputs `y_true` and `y_hat` in alternating rows
        as a single HWC image.
    """

    nrows = y_true.shape[0] // cols

    rows = []
    for _ in range(nrows):
        rows.append(torch.cat(list(y_true[:cols]), dim=1))
        rows.append(torch.cat(list(y_hat[:cols]), dim=1))
        y_true = y_true[cols:]
        y_hat = y_hat[cols:]
    grid = torch.cat(rows, dim=0).cpu().numpy()
    return matplotlib.colormaps[cmap](grid)[..., :3]
