"""Generic RGB image rendering."""

import matplotlib
import numpy as np
import torch
from einops import rearrange
from jaxtyping import Shaped, UInt8
from torch import Tensor


def _normalize(
    data: Shaped[Tensor, "..."], quant: int = 256
) -> UInt8[Tensor, "..."]:
    """Normalize data to `[0, quant]` using its min/max."""
    assert quant <= 256

    if data.dtype == torch.bool:
        return torch.where(data, torch.tensor(quant - 1, dtype=torch.uint8), 0)
    else:
        left = torch.min(data)
        right = torch.max(data)
        return ((quant - 1) * torch.clip(
            (data - left) / (right - left), 0.0, 1.0)).to(torch.uint8)


def tile_images(
    y_true: Shaped[Tensor, "b h w 3"] | Shaped[Tensor, "b h w"],
    *y_hat: Shaped[Tensor, "b h w 3"] | Shaped[Tensor, "b h w"],
    cols: int = 8, cmap: str | Shaped[np.ndarray, "H W 3"] = 'viridis',
    normalize: bool = False
) -> UInt8[np.ndarray, "h2 w2 3"]:
    """Tile a batch of images into a single image as a grid.

    !!! info

        Colors can be specified as the `str` name of a [matplotlib colormap](
        https://matplotlib.org/stable/users/explain/colors/colormaps.html)
        (with the alpha channel discarded) or as a `UInt[N, 3]` numpy array of
        colors.

    !!! note

        If the supplied `y_true` and `y_hat` already have RGB channels
        (`batch x H x W x 3`), the colormap is skipped.

    Args:
        y_true: images to tile. Can be 3-channel RGB or numerical values.
        y_hat: optional secondary images to tile; if provided, each `y_hat`
            is placed under its corresponding `y_true`.
        cols: number of columns. If the number of images does not divide
            `batch`, they are zero padded in the resulting image unless there
            are fewer images than `batch`.
        cmap: colormap to apply, either as the name of a matplotlib colormap
            or a list of colors.
        normalize: whether to normalize the input data. All data is normalized
            and clipped; `y_true` and `y_hat` are normalized separately.
            If `normalize=False`, the caller is expected to provide
            pre-normalized data in `[0.0, 1.0]`.

    Returns:
        Tiled grid of the inputs as a single HWC-format RGB image.
    """
    if normalize:
        if isinstance(cmap, str):
            N = matplotlib.colormaps[cmap].N
        else:
            N = cmap.shape[0]

        y_true = _normalize(y_true, quant=N)
        _y_hat = [_normalize(y, quant=N) for y in y_hat]
    else:
        _y_hat = list(y_hat)

    if len(y_hat) == 0:
        images = y_true
    else:
        images = torch.concatenate([y_true] + _y_hat, dim=1)

    if images.shape[0] < cols:
        cols = images.shape[0]
    elif images.shape[0] % cols != 0:
        padding = torch.zeros(
            [cols - (images.shape[0] % cols), *images.shape[1:]],
            dtype=images.dtype, device=images.device)
        images = torch.concatenate([images, padding], dim=0)

    images_np = images.cpu().numpy()
    if len(images.shape) == 3:
        if isinstance(cmap, str):
            images_np = matplotlib.colormaps[cmap](
                images_np, bytes=True)[..., :3]
        else:
            images_np = cmap[images_np]

    tiled = rearrange(
        images_np, "(rows cols) h w c -> (rows h) (cols w) c",
        cols=cols, rows=images_np.shape[0] // cols)
    return tiled


def swap_angular_conventions(
    data: Shaped[Tensor, "batch h w *c"]
) -> Shaped[Tensor, "batch w h *c"]:
    """Swap image and spatial angular conventions.

    - Image conventions are in elevation-azimuth order (vertical axis is first),
      where increasing elevation is down and increasing azimuth is right.
    - Spatial conventions are in azimuth-elevation order (horizontal axis is
      first) where increasing elevation is up and increasing azimuth is left.

    !!! note

        This function is really just a suggestively-named equivalent to
        ```
        torch.transpose(torch.flip(..., dim=[1, 2]), 1, 2)
        ```
    """
    return torch.transpose(torch.flip(data, dims=[1, 2]), 1, 2)
