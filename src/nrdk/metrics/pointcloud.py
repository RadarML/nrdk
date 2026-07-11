"""Point cloud metrics."""

from abc import ABC, abstractmethod

import numpy as np
import torch
from beartype.typing import Literal
from jaxtyping import Bool, Float, Shaped
from torch import Tensor


class PointCloudMetric(ABC):
    """Generic point-cloud objective (e.g. Chamfer, Hausdorff).

    !!! note

        This base class only supplies the point cloud framework;
        inheritors must implement a `as_points` method, which specifies the
        conversion from the raw inputs to cartesian points.

    !!! tip

        Distance computation is accelerated with `torch_geometric` if installed,
        with a naive brute-force fallback otherwise. Use `require_knn=True` to
        ensure that the optimized implementation is used (and raise an error if
        unavailable).

    Supported modes:

    - `chamfer` (default): chamfer distance (mean)
    - `hausdorff`: hausdorff distance (max)
    - `modhausdorff`: modified hausdorff distance (median)

    Args:
        mode: specified modes.
        on_empty: value to use if one of the provided maps is completely empty.
        require_knn: if `True`, raise an `ImportError` if `torch_geometric`
            (used to accelerate nearest-neighbor lookups) is not installed.
    """

    def __init__(
        self, on_empty: float = 64.0,
        mode: Literal["chamfer", "hausdorff", "modhausdorff"] = "chamfer",
        require_knn: bool = False,
    ) -> None:
        self.mode = mode
        self.on_empty = on_empty

        try:
            from torch_geometric.nn.pool import knn  # type: ignore
            self._knn = knn
        except ImportError:
            self._knn = None
            if require_knn:
                raise ImportError(
                    "torch_geometric is required (require_knn=True) but is "
                    "not installed.") from None

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
        x2 = torch.sum(x * x, dim=-1, keepdim=True)  # [n1, 1]
        y2 = torch.sum(y * y, dim=-1, keepdim=True)  # [n2, 1]
        sq = x2 - 2 * (x @ y.T) + y2.T
        return torch.sqrt(torch.clamp(sq, min=0))

    def _nearest_dists(
        self, pts_x: Float[Tensor, "n1 d"], pts_y: Float[Tensor, "n2 d"]
    ) -> tuple[Float[Tensor, "n2"], Float[Tensor, "n1"]]:
        """Compute per-point nearest-neighbor distances in both directions.

        Returns:
            d1: for each point in `pts_y`, distance to its nearest neighbor
                in `pts_x`.
            d2: for each point in `pts_x`, distance to its nearest neighbor
                in `pts_y`.
        """
        if self._knn is not None:
            # knn(x, y, k) finds, for each element in y, the k nearest points
            # in x; it returns [query_idx (into y), match_idx (into x)].
            idx_y, idx_x = self._knn(pts_x, pts_y, k=1)
            d1 = torch.linalg.norm(pts_y[idx_y] - pts_x[idx_x], dim=-1)
            idx_x2, idx_y2 = self._knn(pts_y, pts_x, k=1)
            d2 = torch.linalg.norm(pts_x[idx_x2] - pts_y[idx_y2], dim=-1)
            return d1, d2
        else:
            dist = self.distance(pts_x, pts_y)
            d1 = torch.min(dist, dim=0).values
            d2 = torch.min(dist, dim=1).values
            return d1, d2

    @torch.compiler.disable
    def _forward(self, x, y):
        pts_x = self.as_points(x)
        pts_y = self.as_points(y)

        if pts_x.shape[0] == 0 or pts_y.shape[0] == 0:
            return torch.full((), self.on_empty, device=x.device)

        d1, d2 = self._nearest_dists(pts_x, pts_y)

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
        _iter = (
            # torch.compiler.disable is type-jank
            self._forward(xs, ys)  # type: ignore
            for xs, ys in zip(y_hat, y_true))
        return torch.stack(list(_iter))


class PolarChamfer2D(PointCloudMetric):
    """2D point cloud metrics for range-azimuth occupancy grids.

    Returns the error in grid units; see [`PointCloudMetric`][^.] for more
    details.

    !!! warning "Not differentiable"

    Args:
        on_empty: value to use if one of the provided maps is completely empty.
        mode: specified modes.
        az_min: minimum azimuth angle in radians.
        az_max: maximum azimuth angle in radians.
        max_points: if set, limit each point cloud to this many points before
            computing distances. Predictions are sampled without replacement
            weighted by sigmoid(logit); ground truth is sampled uniformly.
        require_knn: if `True`, raise an `ImportError` if `torch_geometric`
            (used to accelerate nearest-neighbor lookups) is not installed,
            instead of silently falling back to a brute-force implementation.
    """

    def __init__(
        self, on_empty: float = 64.0,
        mode: Literal["chamfer", "hausdorff", "modhausdorff"] = "chamfer",
        az_min: float = -np.pi / 2, az_max: float = np.pi / 2,
        max_points: int | None = None, require_knn: bool = False,
    ) -> None:
        super().__init__(on_empty=on_empty, mode=mode, require_knn=require_knn)
        self.az_min = az_min
        self.az_max = az_max
        self.max_points = max_points

    @staticmethod
    def _limit(data: Tensor, max_points: int | None) -> Bool[Tensor, "..."]:
        """Subsample occupied cells to at most max_points."""
        occupied = data if data.dtype == torch.bool else data > 0

        if max_points is None:
            return occupied

        occ_idx = occupied.nonzero(as_tuple=False)  # [n, ndim]
        if len(occ_idx) <= max_points:
            return occupied

        if data.dtype == torch.bool:
            weights = torch.ones(len(occ_idx), device=data.device)
        else:
            weights = torch.sigmoid(data[tuple(occ_idx.T)])
        sampled = torch.multinomial(weights, max_points, replacement=False)
        keep = occ_idx[sampled]
        result = torch.zeros_like(occupied)
        result[tuple(keep.T)] = True

        return result

    def as_points(
        self, data: Shaped[Tensor, "azimuth range"]
    ) -> Float[Tensor, "n 2"]:
        """Convert (azimuth, range) occupancy grid to points."""
        data = self._limit(data, self.max_points)
        Na, Nr = data.shape
        az_angles = torch.linspace(
            self.az_min, self.az_max, Na, device=data.device)

        az, r = torch.where(data)
        x = torch.cos(az_angles)[az] * (r + 1)
        y = torch.sin(az_angles)[az] * (r + 1)
        return torch.stack([x, y], dim=-1)


class PolarChamfer3D(PointCloudMetric):
    """3D point cloud metrics for elevation-azimuth depth maps.

    Returns the error in grid units; see [`PointCloudMetric`][^.] for more
    details.

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
        max_points: if set, limit each point cloud to this many points before
            computing distances. Predictions are sampled without replacement
            weighted by sigmoid(logit); ground truth is sampled uniformly.
        require_knn: if `True`, raise an `ImportError` if `torch_geometric`
            (used to accelerate nearest-neighbor lookups) is not installed,
            instead of silently falling back to a brute-force implementation.
    """

    def __init__(
        self, on_empty: float = 64.0,
        mode: Literal["chamfer", "hausdorff", "modhausdorff"] = "chamfer",
        az_min: float = -np.pi / 2, az_max: float = np.pi / 2,
        el_min: float = -np.pi / 4, el_max: float = np.pi / 4,
        max_points: int | None = None, require_knn: bool = False,
    ) -> None:
        super().__init__(on_empty=on_empty, mode=mode, require_knn=require_knn)
        self.az_min = az_min
        self.az_max = az_max
        self.el_min = el_min
        self.el_max = el_max
        self.max_points = max_points

    @staticmethod
    def _limit(data: Tensor, max_points: int | None) -> Bool[Tensor, "..."]:
        """Subsample occupied cells to at most max_points."""
        occupied = data if data.dtype == torch.bool else data > 0

        if max_points is None:
            return occupied

        occ_idx = occupied.nonzero(as_tuple=False)  # [n, ndim]
        if len(occ_idx) <= max_points:
            return occupied

        if data.dtype == torch.bool:
            weights = torch.ones(len(occ_idx), device=data.device)
        else:
            weights = torch.sigmoid(data[tuple(occ_idx.T)])
        sampled = torch.multinomial(weights, max_points, replacement=False)
        keep = occ_idx[sampled]
        result = torch.zeros_like(occupied)
        result[tuple(keep.T)] = True

        return result

    def as_points(
        self, data: Shaped[Tensor, "elevation azimuth range"]
    ) -> Float[Tensor, "n 3"]:
        """Convert (elevation, azimuth, range) occupancy grid to points."""
        data = self._limit(data, self.max_points)
        Ne, Na, _Nr = data.shape
        el_angles = torch.linspace(
            self.el_max, self.el_min, Ne, device=data.device)
        az_angles = torch.linspace(
            self.az_min, self.az_max, Na, device=data.device)

        el, az, rng = torch.where(data)
        rng = rng.float() + 1
        _el_cos = torch.cos(el_angles)[el]
        x = _el_cos * torch.cos(az_angles)[az] * rng
        y = _el_cos * torch.sin(az_angles)[az] * rng
        z = torch.sin(el_angles)[el] * rng

        return torch.stack([x, y, z], dim=-1)
