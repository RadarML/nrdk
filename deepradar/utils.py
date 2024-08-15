"""Miscellaneous utilities."""

import torch
from torch import Tensor
from jaxtyping import Shaped


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
