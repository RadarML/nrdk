"""Reusable voxel rendering utilities."""

import warnings
from typing import Literal, overload

import numpy as np
import torch
from einops import rearrange, reduce
from jaxtyping import Bool, Float, Int16, Shaped, UInt8
from torch import Tensor


def bev_from_polar2(
    data: Shaped[Tensor, "batch azimuth range channels"],
    size: int = 512, theta_min: float = -np.pi, theta_max: float = np.pi
) -> Shaped[Tensor, "batch height width channels"]:
    """Convert 2D polar image to a birds-eye-view cartesian image.

    Uses nearest-neighbor interpolation for a specified resolution.

    Args:
        data: batched polar grid in batch-azimuth-range-channels order; the
            range and azimuth axes should be in "increasing" order (smallest
            range bin / azimuth angle first). Can be any data type (e.g.
            `float` intensity image or `bool` occupancy grid).
        size: desired resolution of the BEV image.
        theta_min: azimuth axis lower bound, with `0` pointing straight up in
            the resulting image, and `+theta` to the left.
        theta_max: azimuth axis upper bound

    Returns:
        Nearest-neighbor-interpolated BEV image, with resolution
            `(resolution, resolution * 2)`. The data type is always preserved.
    """
    rmax = data.shape[2] - 1
    thetamax = data.shape[1] - 1

    x, y = torch.meshgrid(
        torch.linspace(-rmax, rmax, size * 2, device=data.device),
        torch.linspace(rmax, 0, size, device=data.device), indexing='xy')
    r = torch.sqrt(x**2 + y**2)
    theta = torch.asin(x / r)

    ir = torch.floor(r).to(dtype=torch.int)
    itheta = torch.floor(
        ((theta - theta_min) / (theta_max - theta_min)) * thetamax
    ).to(dtype=torch.int)

    # Handle out-of-bounds values in the corners of the image
    mask = (ir > rmax) | (itheta < 0) | (itheta > thetamax)
    ir[mask] = 0
    itheta[mask] = 0
    bev = data[:, itheta, ir]
    bev[:, mask] = 0

    return bev


@overload
def bev_height_from_polar_occupancy(
    data: Bool[Tensor, "batch elevation azimuth range"],
    size: int = 512, theta_min: float = -np.pi, theta_max: float = np.pi,
    scale: Literal[True] = True
) -> Float[Tensor, "batch height width"]: ...

@overload
def bev_height_from_polar_occupancy(
    data: Bool[Tensor, "batch elevation azimuth range"],
    size: int = 512, theta_min: float = -np.pi, theta_max: float = np.pi,
    scale: bool = False
) -> (
    Int16[Tensor, "batch height width"] | UInt8[Tensor, "batch height width"]
): ...

def bev_height_from_polar_occupancy(
    data: Bool[Tensor, "batch elevation azimuth range"],
    size: int = 512, theta_min: float = -np.pi, theta_max: float = np.pi,
    scale: bool = False
) -> (
    Float[Tensor, "batch height width"]
    | Int16[Tensor, "batch height width"] | UInt8[Tensor, "batch height width"]
):
    """Convert azimuth-polar occupancy grid to a birds-eye-view height map.

    Possible return types:

    - If `scale = True`: returns a 0-1 float height map.
    - If `scale = False`:

        - If `n_elevation <= 256`: returns an unsigned 8-bit height map.
        - Otherwise: returns a signed 16-bit height map (pytorch does not
          support unsigned 16-bits).

    !!! warning

        The input data is assumed to already be in an image convention, i.e.
        increasing elevation bins are down, and increasing azimuth bins to
        the right.

    Args:
        data: azimuth-polar (cylindrical or spherical) occupancy grid.
        size: desired resolution of the BEV image.
        theta_min: azimuth axis lower bound, with `0` pointing straight up in
            the resulting image, and `+theta` to the left.
        theta_max: azimuth axis upper bound
        scale: return a scaled (`float`) height map.

    Returns:
        Height map, with resolution `(size, size * 2)`.
    """
    elev_as_channels = rearrange(data, "b el az rng -> b az rng el")
    cartesian = bev_from_polar2(
        elev_as_channels, size=size, theta_min=theta_min, theta_max=theta_max)
    height = torch.argmax(cartesian.to(torch.uint8), dim=-1)

    if scale:
        zmin = reduce(height, "batch height width -> batch", "min")
        zmax = reduce(height, "batch height width -> batch", "max")
        return (height - zmin) / (zmax - zmin)
    else:
        if height.shape[-1] <= 256:
            return height.to(torch.uint8)
        else:
            return height.to(torch.int16)


def depth_from_polar_occupancy(
    data: Bool[Tensor, "batch elevation azimuth range"],
    size: tuple[int, int] | None = None
) -> UInt8[Tensor, "batch height width"] | Int16[Tensor, "batch height width"]:
    """Convert azimuth-polar occupancy grid to a depth map.

    !!! warning

        The input data is assumed to already be in an image convention, i.e.
        increasing elevation bins are down, and increasing azimuth bins to
        the right.

    !!! bug

        Pytorch `nn.functional.interpolate` doesn't support int16; a manual
        fallback will need to be implemented to support `n_range > 256` and
        `size != None`.

    Args:
        data: azimuth-polar (cylindrical or spherical) occupancy grid.
        size: `(height, width)` of the output map; up/down-sampled using
            nearest neighbor interpolation. If `None`, no resizing is
            performed.

    Returns:
        Integer depth map, as `uint8` if `n_range <= 256` and `int16`
            otherwise.
    """
    depth = torch.argmax(data.to(torch.uint8), dim=-1)
    if data.shape[-1] <= 256:
        depth = depth.to(torch.uint8)
    else:
        depth = depth.to(torch.int16)

    if size is not None:
        if depth.dtype is not torch.uint8:
            warnings.warn(
                "Since there are more than 256 range bins, depth is "
                "computed as `int16`; however, "
                "torch.nn.functional.interpolate might not support int16!")
        depth = torch.nn.functional.interpolate(
            depth[:, None, :, :], size=size, mode='nearest')[:, 0, :, :]

    return depth
