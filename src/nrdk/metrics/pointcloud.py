"""Point cloud metrics."""

from abc import ABC, abstractmethod

import numpy as np
import torch
from beartype.typing import Literal
from jaxtyping import Bool, Float, Num, Shaped
from torch import Tensor


class PointCloudMetric(ABC):
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

    def __init__(
        self, on_empty: float = 64.0,
        mode: Literal["chamfer", "hausdorff", "modhausdorff"] = "chamfer"
    ) -> None:
        self.mode = mode
        self.on_empty = on_empty

    @abstractmethod
    def as_points(self, data: Shaped[Tensor, "..."]) -> Float[Tensor, "n d"]:
        """Convert a model's native representation to cartesian points.

        Note that since the output number of points may vary across a batch,
        this method is not batched.
        """
        ...

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
        else:  # "hausdorff"
            return torch.max(torch.concatenate([d1, d2]))

    def __call__(
        self, y_hat: Shaped[Tensor, "..."], y_true: Shaped[Tensor, "..."],
    ) -> Float[Tensor, "batch"]:
        """Compute point cloud metric."""
        _iter = (self._forward(xs, ys) for xs, ys in zip(y_hat, y_true))
        return torch.stack(list(_iter))


class PolarChamfer2D(PointCloudMetric):
    """2D point cloud metrics for range-azimuth occupancy grids.

    Returns the error in grid units.

    !!! warning "Not differentiable"

    Args:
        on_empty: value to use if one of the provided maps is completely empty.
        mode: specified modes.
        az_min: minimum azimuth angle in radians.
        az_max: maximum azimuth angle in radians.
    """

    def __init__(
        self, on_empty: float = 64.0,
        mode: Literal["chamfer", "hausdorff", "modhausdorff"] = "chamfer",
        az_min: float = -np.pi / 2, az_max: float = np.pi / 2
    ) -> None:
        super().__init__(on_empty=on_empty, mode=mode)
        self.az_min = az_min
        self.az_max = az_max

    def as_points(
        self, data: Bool[Tensor, "azimuth range"]
    ) -> Float[Tensor, "n 2"]:
        """Convert (azimuth, range) occupancy grid to points."""
        Na, Nr = data.shape
        az_angles = torch.linspace(
            self.az_min, self.az_max, Na, device=data.device)

        az, r = torch.where(data)
        x = torch.cos(az_angles)[az] * r
        y = torch.sin(az_angles)[az] * r
        return torch.stack([x, y], dim=-1)


class PolarChamfer3D(PointCloudMetric):
    """3D point cloud metrics for elevation-azimuth depth maps.

    Returns the error in grid units.

    !!! warning "Image Axis Order"

        The elevation axis is in an image convention, where increasing index
        corresponds to a lower elevation angle / downward direction.

    !!! warning "Not Differentiable"

    Args:
        on_empty: value to use if one of the provided maps is completely empty.
        mode: specified modes.
        az_min: minimum azimuth angle in radians.
        az_max: maximum azimuth angle in radians.
        el_min: minimum elevation angle in radians.
        el_max: maximum elevation angle in radians.
    """

    def __init__(
        self, on_empty: float = 64.0,
        mode: Literal["chamfer", "hausdorff", "modhausdorff"] = "chamfer",
        az_min: float = -np.pi / 2, az_max: float = np.pi / 2,
        el_min: float = -np.pi / 4, el_max: float = np.pi / 4
    ) -> None:
        super().__init__(on_empty=on_empty, mode=mode)
        self.az_min = az_min
        self.az_max = az_max
        self.el_min = el_min
        self.el_max = el_max

    def as_points(
        self, data: Num[Tensor, "elevation azimuth"]
    ) -> Float[Tensor, "n 3"]:
        """Convert elevation-azimuth depth maps to points."""
        Ne, Na = data.shape
        el_angles = torch.linspace(
            self.el_max, self.el_min, Ne, device=data.device)
        az_angles = torch.linspace(
            self.az_min, self.az_max, Na, device=data.device)

        el, az = torch.where(data)
        _el_cos = torch.cos(el_angles)[el]
        x = _el_cos * torch.cos(az_angles)[az] * data[el, az]
        y = _el_cos * torch.sin(az_angles)[az] * data[el, az]
        z = torch.sin(el_angles)[el] * data[el, az]

        return torch.stack([x, y, z], dim=-1)
