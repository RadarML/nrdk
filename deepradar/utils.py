"""Miscellaneous utilities."""

import matplotlib
import numpy as np
import torch
from beartype.typing import Literal
from jaxtyping import Bool, Float, Num, Shaped
from torch import Tensor


def polar2_to_bev(
    data: Shaped[Tensor, "batch azimuth range"], height: int = 512
) -> Shaped[Tensor, "batch height width"]:
    """Convert 2D polar image to a birds-eye-view cartesian image.

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


def polar3_to_bev(
    data: Bool[Tensor, "batch elevation azimuth range"],
    az_span: float = np.pi / 2, el_span: float = np.pi / 4,
    mode: Literal['lowest', 'highest'] = 'highest'
) -> Shaped[Tensor, "batch x y"]:
    """Convert 3D polar grid to a birds-eye-view cartesian height map.

    Return values indicate the highest (or lowest) point in each grid cell,
    with `1` being the highest/lowest point, `n` being `n - 1` units below the
    highest, and `0` indicating grid cells with no points.

    Args:
        data: batched polar grid in batch-elevation-azimuth-range order.
            Must be a boolean type, with `True` indicating occupied grids.
        az_span, el_span: angle, in radians, spanned by the elevation
            and azimuth axes in the input image. If these values exceed
            `pi/2`, some points may be cropped.
        mode: compute lowest or highest height.

    Returns:
        Batched BEV image, with spatial resolution equal to the range
        resolution.
    """
    nr = data.shape[-1]
    el_angles = torch.linspace(el_span, -el_span, nr, device=data.device)
    az_angles = torch.linspace(-az_span, az_span, nr * 2, device=data.device)

    res = []
    for polar in data:
        el, az, rng = torch.where(polar)

        _el_cos = torch.cos(el_angles)[el]

        x = _el_cos * torch.cos(az_angles)[az] * rng
        y = _el_cos * torch.sin(az_angles)[az] * rng
        z = torch.sin(el_angles)[el] * rng
        if mode == 'highest':
            z = -z

        ix = torch.clip(x.to(dtype=torch.int), min=0, max=nr - 1)
        iy = torch.clip(y.to(dtype=torch.int) + nr, min=0, max=nr * 2)
        iz = torch.clip(z.to(dtype=torch.int) + nr, 0, nr * 2 - 1)

        # Special case for empty: just give a blank result
        if iz.shape[0] == 0:
            res.append(torch.zeros(nr, nr * 2, dtype=torch.uint8))
        else:
            zmin = torch.min(iz) - 1
            zmax = 127 - torch.max(iz)

            cartesian = torch.zeros(
                nr, nr * 2, int(128 - zmin - zmax),
                dtype=torch.uint8, device=data.device)
            cartesian[ix, iy, iz - zmin] = 1

            height = -torch.argmax(cartesian, dim=2)
            res.append(torch.where(height == 0, 0, height - zmin))

    return torch.stack(res)


def _normalize(data: Shaped[Tensor, "..."]) -> Float[Tensor, "..."]:
    """Normalize data to [0, 1] using its min/max."""
    left = torch.min(data)
    right = torch.max(data)
    return torch.clip((data - left) / (right - left), 0.0, 1.0)


def comparison_grid(
    y_true: Shaped[Tensor, "batch h w"], y_hat: Shaped[Tensor, "batch h w"],
    cols: int = 8, cmap: str = 'viridis', normalize: bool = False
) -> Num[np.ndarray, "h2 w2 3"]:
    """Create image comparison grid.

    Args:
        y_true, y_hat: images to compare; nominally y_true/y_hat, though the
            exact order does not matter.
        cols: number of columns. Any excess images (which do not fill a full
            row) will be discarded.
        cmap: matplotlib colormap to use, e.g. `viridis`, `inferno`. Note that
            the alpha channel is discarded.
        normalize: whether to normalize the input data. All data is normalized
            and clipped; `y_true` and `y_hat` are normalized separately.
            If `normalize=False`, the caller is expected to provide
            pre-normalized data in `[0.0, 1.0]`.

    Returns:
        Colormapped grid of the inputs `y_true` and `y_hat` in alternating rows
        as a single HWC image.
    """
    if normalize:
        y_true = _normalize(y_true)
        y_hat = _normalize(y_hat)

    nrows = y_true.shape[0] // cols
    rows = []
    for _ in range(nrows):
        rows.append(torch.cat(list(y_true[:cols]), dim=1))
        rows.append(torch.cat(list(y_hat[:cols]), dim=1))
        y_true = y_true[cols:]
        y_hat = y_hat[cols:]
    grid = torch.cat(rows, dim=0).cpu().numpy()
    return matplotlib.colormaps[cmap](grid)[..., :3]
