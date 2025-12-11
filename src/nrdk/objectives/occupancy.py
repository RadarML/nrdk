"""Radar training objectives and common building blocks for losses/metrics."""

from collections.abc import Mapping
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np
import torch
from abstract_dataloader.ext.objective import Objective, VisualizationConfig
from einops import rearrange, reduce
from jaxtyping import Bool, Float, Shaped
from torch import Tensor

from nrdk import vis
from nrdk.metrics import (
    BCE,
    BinaryDiceLoss,
    PolarChamfer2D,
    PolarChamfer3D,
    VoxelDepth,
)


@runtime_checkable
class Occupancy3DData(Protocol):
    """Protocol type for 3D occupancy data.

    !!! warning

        The data are nominally in "image conventions": increasing elevation
        is down and increasing azimuth is to the right (increasing range
        is away).

    Attributes:
        occupancy: occupancy grid, with batch-spatial axis order.
    """

    occupancy: Bool[Tensor, "batch t elevation azimuth range"]


@runtime_checkable
class Occupancy2DData(Protocol):
    """Protocol type for 2D occupancy data.

    !!! warning

        The data are nominally in "image conventions": increasing azimuth
        is to the right (and increasing range is away).

    Attributes:
        occupancy: occupancy grid, with batch-range-azimuth axis order.
    """

    occupancy: Bool[Tensor, "batch t azimuth range"]


class Occupancy3D(Objective[
    Tensor, Occupancy3DData, Float[Tensor, "batch t elevation azimuth range"]]
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

    ??? quote "Sample Hydra Config"

        ```yaml title="objectives/lidar3d.yaml"
        occupancy3d:
          weight: 1.0
          y_true: occ3d
          y_pred: occ3d
          objective:
            _target_: nrdk.objectives.Occupancy3D
            range_weighted: True
            positive_weight: 64.0
            mode: spherical
            vis_config:
            cols: 8
            width: 512
            height: 256
            cmaps:
              bev: inferno
              depth: viridis
        ```

    Args:
        range_weighted: Whether to apply equal-area "range weighting", where
            the relative weight of bins is adjusted on the range axis so that
            each bin is weighted according to the area which it represents
            (i.e. multiplying the weight by `range`).
        positive_weight: weight to give occupied samples to account for
            sparsity / class imbalance.
        max_range: value to assign to chamfer loss when there are no points;
            nominally the number of range bins.
        az_min: minimum azimuth angle.
        az_max: maximum azimuth angle.
        el_min: minimum elevation angle.
        el_max: maximum elevation angle.
        vis_config: visualization configuration; the `cmaps` can have `bev`
            and `depth` keys.
    """

    def __init__(
        self, range_weighted: bool = True, positive_weight: float = 16.0,
        max_range: float = 64.0,
        mode: Literal["spherical", "cylindrical"] = "spherical",
        az_min: float = -np.pi / 2, az_max: float = np.pi / 2,
        el_min: float = -np.pi / 4, el_max: float = np.pi / 4,
        vis_config: VisualizationConfig | Mapping[str, Any] = {},
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
            el_min=el_min, el_max=el_max, on_empty=max_range)

        if not isinstance(vis_config, VisualizationConfig):
            vis_config = VisualizationConfig(**vis_config)
        self.vis_config = vis_config

    def __call__(
        self, y_true: Occupancy3DData,
        y_pred: Float[Tensor, "batch t elevation azimuth range"],
        train: bool = True
    ) -> tuple[Float[Tensor, "batch"], dict[str, Float[Tensor, "batch"]]]:
        _y_pred = rearrange(y_pred, "b t el az rng -> (b t) el az rng")
        occ_hat = _y_pred > 0
        occ_true = rearrange(
            y_true.occupancy, "b t el az rng -> (b t) el az rng")

        metrics = {
            "bce": self.bce(occ_true, _y_pred),
            "height": self.height(occ_true, occ_hat),
            "depth": self.depth(occ_true, occ_hat)
        }
        if not train:
            depth_true = torch.argmax(occ_true.to(torch.uint8), dim=-1)
            depth_hat = torch.argmax(occ_hat.to(torch.uint8), dim=-1)
            metrics["chamfer"] = self.chamfer(depth_hat, depth_true)

        # Temporal reduction
        metrics = {
            k: reduce(
                v, "(b t) -> b", "mean", b=y_pred.shape[0], t=y_pred.shape[1])
            for k, v in metrics.items()}
        return metrics["bce"], metrics

    def visualizations(
        self, y_true: Occupancy3DData,
        y_pred: Float[Tensor, "batch t elevation azimuth range"]
    ) -> dict[str, Shaped[np.ndarray, "H W 3"]]:
        occ = y_pred[:, -1] > 0
        gt = y_true.occupancy[:, -1]

        bev_true, bev_hat = (
            vis.bev_height_from_polar_occupancy(
                y, size=self.vis_config.height, theta_min=self.theta_min,
                theta_max=self.theta_max, scale=False)
            for y in (gt, occ))
        bev_vis = vis.tile_images(
            bev_true, bev_hat, cols=self.vis_config.cols,
            cmap=self.vis_config.cmaps.get("bev", "inferno"), normalize=True)

        depth_true, depth_hat = (
            vis.depth_from_polar_occupancy(
                y, size=(self.vis_config.height, self.vis_config.width))
            for y in (gt, occ))
        depth_vis = vis.tile_images(
            depth_true, depth_hat, cols=self.vis_config.cols,
            cmap=self.vis_config.cmaps.get("depth", "viridis"), normalize=True)

        return {"bev": bev_vis, "depth": depth_vis}

    def render(
        self, y_true: Occupancy3DData,
        y_pred: Float[Tensor, "batch t elevation azimuth range"],
        render_gt: bool = False
    ) -> dict[str, Shaped[np.ndarray, "batch ..."]]:
        occ = y_pred[:, -1] > 0

        rendered = {"occ3d": occ.to(torch.uint8).cpu().numpy()}
        if render_gt:
            rendered["occ3d_gt"] = y_true.occupancy.to(
                torch.uint8).cpu().numpy()
        return rendered


class Occupancy2D(Objective[
    Tensor, Occupancy2DData, Float[Tensor, "batch t range azimuth"]
]):
    """2D polar occupancy prediction objective.

    This objective uses the same mixed BCE + Dice loss formulation as
    [RadarHD](https://arxiv.org/abs/2206.09273).

    Metrics:
        - `loss`: weighted BCE & Dice loss.
        - `bce`: Binary Cross Entropy loss, with range weighting if specified.
        - `dice`: Dice loss, also with range weighting.
        - `chamfer`: 2D point cloud Chamfer distance.

    Visualizations:
        - `bev`: Bird's Eye View occupancy grid.

    ??? quote "Sample Hydra Config"

        ```yaml title="objectives/lidar2d.yaml"
        occupancy2d:
          weight: 1.0
          y_true: occ2d
          y_pred: occ2d
          objective:
            _target_: nrdk.objectives.Occupancy2D
            range_weighted: True
            positive_weight: 64.0
            bce_weight: 0.9
            vis_config:
            cols: 8
            width: 512
            height: 256
            cmaps:
              bev: inferno
        ```

    Args:
        range_weighted: Whether to apply equal-area "range weighting", where
            the relative weight of bins is adjusted on the range axis so that
            each bin is weighted according to the area which it represents
            (i.e. multiplying the weight by `range`).
        positive_weight: weight to give occupied samples; nominally the number
            of range bins.
        bce_weight: BCE loss weight; Dice loss is weighted `1 - bce_weight`.
        az_min: minimum azimuth angle.
        az_max: maximum azimuth angle.
        vis_config: visualization configuration; `cmaps` can have a `bev` key.
    """

    def __init__(
        self, range_weighted: bool = True, positive_weight: float = 64.0,
        bce_weight: float = 0.9,
        az_min: float = -np.pi / 2, az_max: float = np.pi / 2,
        vis_config: VisualizationConfig | Mapping = {},
    ) -> None:
        self.az_min = az_min
        self.az_max = az_max
        self.bce_weight = bce_weight

        self.bce_loss = BCE(
            positive_weight=positive_weight,
            weighting='cylindrical' if range_weighted else None)
        self.dice_loss = BinaryDiceLoss(
            weighting='cylindrical' if range_weighted else None)
        self.chamfer = PolarChamfer2D(
            mode="chamfer", az_min=az_min, az_max=az_max)

        if not isinstance(vis_config, VisualizationConfig):
            vis_config = VisualizationConfig(**vis_config)
        self.vis_config = vis_config

    def __call__(
        self, y_true: Occupancy2DData,
        y_pred: Float[Tensor, "batch t azimuth range"],
        train: bool = True
    ) -> tuple[Float[Tensor, "batch"], dict[str, Float[Tensor, "batch"]]]:
        _y_pred = rearrange(y_pred, "b t az rng -> (b t) az rng")
        occ_true = rearrange(y_true.occupancy, "b t az rng -> (b t) az rng")

        metrics = {
            # bce_loss takes [batch * azimuth range]
            "bce": self.bce_loss(
                occ_true[:, None, :, :], _y_pred[:, None, :, :]),
            "dice": self.dice_loss(
                occ_true[:, None, :, :], _y_pred[:, None, :, :])
        }
        if not train:
            metrics["chamfer"] = self.chamfer(_y_pred > 0, occ_true)

        # Temporal reduction
        metrics = {
            k: reduce(
                v, "(b t) -> b", "mean", b=y_pred.shape[0], t=y_pred.shape[1])
            for k, v in metrics.items()}
        loss = (
            metrics["bce"] * self.bce_weight
            + metrics["dice"] * (1 - self.bce_weight))
        return loss, metrics

    def visualizations(
        self, y_true: Occupancy2DData,
        y_pred: Float[Tensor, "batch t azimuth range"]
    ) -> dict[str, Shaped[np.ndarray, "H W 3"]]:
        occ = torch.nn.functional.sigmoid(y_pred)[:, -1]

        bev_true, bev_hat = (
            # There's an extra channels axis that we bypass
            vis.voxels.bev_from_polar2(
                y[..., None], size=self.vis_config.height,
                theta_min=self.az_min, theta_max=self.az_max)[..., 0]
            for y in (y_true.occupancy[:, -1], occ))
        bev_vis = vis.tile_images(
            bev_true, bev_hat, cols=self.vis_config.cols,
            cmap=self.vis_config.cmaps.get("bev", "inferno"), normalize=True)

        return {"bev": bev_vis}

    def render(
        self, y_true: Occupancy3DData,
        y_pred: Float[Tensor, "batch t elevation azimuth range"],
        render_gt: bool = False
    ) -> dict[str, Shaped[np.ndarray, "batch ..."]]:
        occ_hat = torch.nn.functional.sigmoid(y_pred)[:, -1]

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
