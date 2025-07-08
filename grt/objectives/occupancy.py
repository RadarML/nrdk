"""Radar training objectives and common building blocks for losses/metrics."""

from functools import partial
from typing import Any, Literal, Mapping, Protocol

import numpy as np
import torch
from abstract_dataloader.ext.objective import Objective, VisualizationConfig
from jaxtyping import Float, Shaped
from torch import Tensor

from grt import vis
from grt.metrics import BCE, PolarChamfer2D, PolarChamfer3D, VoxelDepth


class Occupancy3DData(Protocol):
    """Protocol type for 3D occupancy data.

    Attributes:
        occupancy: occupancy grid, with batch-spatial axis order.
    """

    occupancy: Float[Tensor, "batch range azimuth elevation"]


class Occupancy2DData(Protocol):
    """Protocol type for 2D occupancy data.

    Attributes:
        occupancy: occupancy grid, with batch-range-azimuth axis order.
    """

    occupancy: Float[Tensor, "batch range azimuth"]


class Occupancy3D(Objective[
    Tensor, Occupancy3DData, Float[Tensor, "batch range azimuth elevation"]]
):
    """3D polar/spherical occupancy prediction objective.

    Metrics:
        - `bce`: weighted BCE loss.
        - `height`: height mean absolute error (in grid units), calculated
            from the occupancy grid.
        - `depth`: depth mean absolute error (in grid units), rendered from the
            occupancy grid.

    Visualizations:
        - `bev`: Bird's Eye View cartesian height map, measured by the highest
            occupied grid cell at each point. Interpolated from the output
            coordinates.
        - `depth`: Rendered depth map.

    Args:
        range_weighted: Whether to apply equal-area "range weighting", where
            the relative weight of bins is adjusted on the range axis so that
            each bin is weighted according to the area which it represents
            (i.e. multiplying the weight by `range`).
        positive_weight: weight to give occupied samples; nominally the number
            of range bins.
        az_min: minimum azimuth angle.
        az_max: maximum azimuth angle.
        el_min: minimum elevation angle.
        el_max: maximum elevation angle.
        vis_config: visualization configuration; the `cmaps` can have `bev`
            and `depth` keys.
    """

    def __init__(
        self, range_weighted: bool = True, positive_weight: float = 64.0,
        mode: Literal["spherical", "cylindrical"] = "cylindrical",
        az_min: float = -np.pi / 2, az_max: float = np.pi / 2,
        el_min: float = -np.pi / 4, el_max: float = np.pi / 4,
        vis_config: VisualizationConfig | Mapping[str, Any] = {}
    ) -> None:
        self.az_min = az_min
        self.az_max = az_max
        self.theta_min = az_min
        self.theta_max = az_max

        self.bce = BCE(
            positive_weight=positive_weight,
            weighting=mode if range_weighted else None)
        self.height = VoxelDepth(axis=2, reverse=True, ord=1)
        self.depth = VoxelDepth(axis=0, reverse=False, ord=1)
        self.chamfer = PolarChamfer3D(
            mode='chamfer', az_min=az_min, az_max=az_max,
            el_min=el_min, el_max=el_max)

        if not isinstance(vis_config, VisualizationConfig):
            vis_config = VisualizationConfig(**vis_config)
        self.vis_config = vis_config

    def __call__(
        self, y_true: Occupancy3DData,
        y_pred: Float[Tensor, "batch range azimuth elevation"],
        train: bool = True
    ) -> tuple[Float[Tensor, "batch"], dict[str, Float[Tensor, "batch"]]]:
        occ = y_pred > 0
        metrics = {
            "bce": self.bce(y_true.occupancy, y_pred),
            "height": self.height(y_true.occupancy, occ),
            "depth": self.depth(y_true.occupancy, occ)
        }
        if not train:
            depth_true = torch.argmax(y_true.occupancy, dim=1)
            depth_hat = torch.argmax(occ, dim=1)
            metrics["chamfer"] = self.chamfer(depth_hat, depth_true)

        return metrics["bce"], metrics

    def visualizations(
        self, y_true: Occupancy3DData,
        y_pred: Float[Tensor, "batch range azimuth elevation"]
    ) -> dict[str, Shaped[np.ndarray, "H W 3"]]:
        occ = y_pred > 0

        bev_true, bev_hat = (
            vis.bev_height_from_polar_occupancy(
                y, size=self.vis_config.height, theta_min=self.theta_min,
                theta_max=self.theta_max, scale=False)
            for y in (y_true.occupancy, occ))
        bev_vis = vis.tile_images(
            bev_true, bev_hat, cols=self.vis_config.cols,
            cmap=self.vis_config.cmaps.get("bev", "inferno"), normalize=True)

        depth_true, depth_hat = (
            vis.depth_from_polar_occupancy(
                y, size=(self.vis_config.height, self.vis_config.width))
            for y in (y_true.occupancy, occ))
        depth_vis = vis.tile_images(
            depth_true, depth_hat, cols=self.vis_config.cols,
            cmap=self.vis_config.cmaps.get("depth", "viridis"), normalize=True)

        return {"bev": bev_vis, "depth": depth_vis}

    def render(
        self, y_true: Occupancy3DData,
        y_pred: Float[Tensor, "batch range azimuth elevation"],
        render_gt: bool = False
    ) -> dict[str, Shaped[np.ndarray, "batch ..."]]:
        occ = y_pred > 0

        rendered = {
            "bev": vis.bev_height_from_polar_occupancy(
                occ, size=self.vis_config.height, theta_min=self.theta_min,
                theta_max=self.theta_max, scale=False),
            "depth": vis.depth_from_polar_occupancy(
                occ, size=(self.vis_config.height, self.vis_config.width))
        }
        if render_gt:
            rendered["bev_gt"] = vis.bev_height_from_polar_occupancy(
                y_true.occupancy, size=self.vis_config.height,
                theta_min=self.theta_min, theta_max=self.theta_max,
                scale=False)
            rendered["depth_gt"] = vis.depth_from_polar_occupancy(
                y_true.occupancy,
                size=(self.vis_config.height, self.vis_config.width))

        return {k: v.cpu().numpy() for k, v in rendered.items()}


class Occupancy2D(Objective[
    Tensor, Occupancy2DData, Float[Tensor, "batch range azimuth"]
]):
    """2D polar occupancy prediction objective.

    Type Parameters:
        - `YTrue`: ground truth data type.
        - `YPred`: model output data type.

    Metrics:
        loss: weighted BCE loss.

    Visualizations:
        `bev`: Bird's Eye View occupancy grid.

    Args:
        range_weighted: Whether to apply equal-area "range weighting", where
            the relative weight of bins is adjusted on the range axis so that
            each bin is weighted according to the area which it represents
            (i.e. multiplying the weight by `range`).
        positive_weight: weight to give occupied samples; nominally the number
            of range bins.
        az_min: minimum azimuth angle.
        az_max: maximum azimuth angle.
        vis_config: visualization configuration; `cmaps` can have a `bev` key.
    """

    def __init__(
        self, range_weighted: bool = True, positive_weight: float = 64.0,
        az_min: float = -np.pi / 3, az_max: float = np.pi / 3,
        vis_config: VisualizationConfig | Mapping = {}
    ) -> None:
        self.az_min = az_min
        self.az_max = az_max
        self.bce_loss = BCE(
            positive_weight=positive_weight,
            weighting='cylindrical' if range_weighted else None)
        self.chamfer = PolarChamfer2D(
            mode="chamfer", az_min=az_min, az_max=az_max)

        if not isinstance(vis_config, VisualizationConfig):
            vis_config = VisualizationConfig(**vis_config)
        self.vis_config = vis_config

    def __call__(
        self, y_true: Occupancy2DData,
        y_pred: Float[Tensor, "batch range azimuth"], train: bool = True
    ) -> tuple[Float[Tensor, "batch"], dict[str, Float[Tensor, "batch"]]]:
        metrics = {
            # bce_loss takes [range y z]
            "bce": self.bce_loss(
                y_true.occupancy[..., None], y_pred[..., None])
        }
        if not train:
            metrics["chamfer"] = self.chamfer(y_pred > 0, y_true.occupancy)

        return metrics["bce"], metrics

    def visualizations(
        self, y_true: Occupancy2DData, y_pred: Float[Tensor, "batch range azimuth"]
    ) -> dict[str, Shaped[np.ndarray, "H W 3"]]:
        occ = torch.nn.functional.sigmoid(y_pred)

        bev_true, bev_hat = (
            # There's an extra channels axis that we bypass
            vis.voxels.bev_from_polar2(
                y[..., None], size=self.vis_config.height,
                theta_min=self.az_min, theta_max=self.az_max)[..., 0]
            for y in (y_true.occupancy, occ))
        bev_vis = vis.tile_images(
            bev_true, bev_hat, cols=self.vis_config.cols,
            cmap=self.vis_config.cmaps.get("bev", "inferno"), normalize=True)

        return {"bev": bev_vis}

    def render(
        self, y_true: Occupancy3DData,
        y_pred: Float[Tensor, "batch range azimuth elevation"],
        render_gt: bool = False
    ) -> dict[str, Shaped[np.ndarray, "batch ..."]]:
        occ_hat = torch.nn.functional.sigmoid(y_pred)

        rendered = {
            "bev":  vis.voxels.bev_from_polar2(
                occ_hat[..., None], size=self.vis_config.height,
                theta_min=self.az_min, theta_max=self.az_max)[..., 0]
        }
        if render_gt:
            rendered["bev_gt"] = vis.voxels.bev_from_polar2(
                y_true.occupancy[..., None], size=self.vis_config.height,
                theta_min=self.az_min, theta_max=self.az_max)[..., 0]

        return {k: v.cpu().numpy() for k, v in rendered.items()}
