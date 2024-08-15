"""RadarHD Objective."""

import numpy as np
import torch
from torch import Tensor
from torch.nn.functional import binary_cross_entropy_with_logits, sigmoid

from jaxtyping import Shaped, Float, Bool
from beartype.typing import cast

from deepradar.utils import polar_to_bev
from .base import MetricValue, Module


class CombinedDiceBCE:
    """Weighted combination of Dice and BCE loss.

    Supports equal-area "range weighting," where the relative weight of bins
    is adjusted on the range axis so that each bin is weighted according to
    the area which it represents (i.e. multiplying the weight by its range).

    Args:
        bce_weight: BCE loss weight; Dice loss is weighted `1 - bce_weight`.
        range_weighted: perform range equal-area weighting if `True`.
    """

    def __init__(
        self, bce_weight: float = 0.9, range_weighted: bool = False
    ) -> None:
        self.bce_weight = bce_weight
        self.range_weighted = range_weighted

    @staticmethod
    def _dice(
        y_hat: Float[Tensor, "batch azimuth range"],
        y_true: Float[Tensor, "batch azimuth range"], weight
    ) -> Float[Tensor, "batch"]:
        denominator = (
            torch.sum(y_hat * y_hat * weight, dim=(1, 2))
            + torch.sum(y_true * weight, dim=(1, 2)))
        numerator = 2 * torch.sum(y_hat * y_true * weight, dim=(1, 2))
        return 1.0 - numerator / denominator

    def __call__(
        self, y_hat: Float[Tensor, "batch azimuth range"],
        y_true: Bool[Tensor, "batch azimuth range"], reduce: bool = True
    ) -> MetricValue:
        """Get Dice + BCE weighted loss.

        Args:
            y_hat: output logits.
            y_true: occupancy grid values.

        Returns:
            Loss value, possibly reduced.
        """
        y_true = y_true.to(y_hat.dtype)

        if self.range_weighted:
            bins = torch.arange(y_true.shape[2], device=y_hat.device)
            weight = ((bins + 1) / y_true.shape[2])[None, None, :]
        else:
            # Mypy doesn't seem to recognize the type overloading here.
            weight = 1.0  # type: ignore

        dice = self._dice(sigmoid(y_hat), y_true, weight)
        bce = torch.mean(
            weight * binary_cross_entropy_with_logits(
                y_hat, y_true, reduction='none'),
            dim=(1, 2))

        loss = bce * self.bce_weight + dice * (1 - self.bce_weight)
        if reduce:
            loss = torch.mean(loss)

        return loss


class Chamfer:
    """L2 Chamfer distance for a polar range-azimuth grid, in range bins.

    Supported modes:

    - `chamfer` (default): chamfer distance (mean)
    - `hausdorff`: hausdorff distance (max)
    - `modhausdorff`: modified hausdorff distance (median)

    Args:
        mode: specified modes.
        on_empty: value to use if one of the provided maps is completely empty.
    """

    def __init__(self, on_empty: float = 64.0, mode: str = "chamfer") -> None:
        self.mode = mode
        self.on_empty = on_empty

    @staticmethod
    def as_points(mask: Bool[Tensor, "azimuth range"]) -> Float[Tensor, "n 2"]:
        """Convert (azimuth, range) occupancy grid to points."""
        bin, r = torch.where(mask)
        phi = (bin - mask.shape[0] // 2) / mask.shape[0] * np.pi
        x = torch.cos(phi) * r
        y = torch.sin(phi) * r
        return torch.stack([x, y]).T

    @staticmethod
    def distance(
        x: Float[Tensor, "n1 d"], y: Float[Tensor, "n2 d"]
    ) -> Float[Tensor, "n1 n2"]:
        """Compute the pairwise distances between x and y."""
        return torch.sqrt(torch.sum(torch.square(
            x[:, None, :] - y[None, :, :]
        ), dim=-1))

    def __call__(
        self, y_hat: Bool[Tensor, "b w h"], y_true: Bool[Tensor, "b w h"],
        reduce: bool = True
    ) -> MetricValue:
        """Compute chamfer distance, in range bins."""

        def _forward(x, y):
            pts_x = self.as_points(x)
            pts_y = self.as_points(y)
            dist = self.distance(pts_x, pts_y)

            if dist.shape[0] == 0 or dist.shape[1] == 0:
                return torch.full((), self.on_empty)

            d1 = torch.min(dist, dim=0).values
            d2 = torch.min(dist, dim=1).values

            if self.mode == "modhausdorff":
                return torch.median(torch.concatenate([d1, d2]))
            elif self.mode == "chamfer":
                return (torch.mean(d1) + torch.mean(d2)) / 2
            else:
                raise ValueError(f"Invalid mode: {self.mode}")

        _iter = (_forward(xs, ys) for xs, ys in zip(y_hat, y_true))

        if reduce:
            return cast(Float[Tensor, ""], sum(_iter)) / y_hat.shape[0]
        else:
            return torch.Tensor(list(_iter))


class RadarHD(Module):
    """Radar -> lidar as bird's eye view (BEV) occupancy.

    Args:
        bce_weight: BCE loss weight; Dice loss is weighted `1 - bce_weight`.
        range_weighted: Whether to apply range weighting; see
            :class:`.CombinedDiceBCE` for details.
        kwargs: see :class:`.BaseModule`.
    """

    STOPPING_METRIC = "loss/val"
    DEFAULT_CMAP = "inferno"

    def __init__(
        self, bce_weight: float = 0.9, range_weighted: bool = True, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.loss = CombinedDiceBCE(
            bce_weight=bce_weight, range_weighted=range_weighted)
        self.chamfer = Chamfer(mode="chamfer")
        self.hausdorff = Chamfer(mode="modhausdorff")
        self.save_hyperparameters()

    def convert_image(
        self, img: Shaped[torch.Tensor, "batch h w"]
    ) -> Shaped[torch.Tensor, "batch h2 w2"]:
        """Convert image for display."""
        return polar_to_bev(img, height=512)

    def training_step(self, batch, batch_idx):  # type: ignore
        """Standard lightning training step."""
        y_hat = self.model(batch['radar'])
        y_true = batch['lidar']
        loss = self.loss(y_hat, y_true)

        self.log(
            "loss/train", loss, on_step=True, on_epoch=True, sync_dist=True)
        if self.global_step % self.log_interval == 0:
            self.log_image_comparison(
                "sample/train", y_true[:self.num_examples],
                sigmoid(y_hat[:self.num_examples].detach()))

        return loss

    def validation_step(self, batch, batch_idx):  # type: ignore
        """Standard lightning validation step."""
        if batch_idx == 0:
            samples = self.trainer.datamodule.val_samples  # type: ignore
            radar = torch.from_numpy(
                samples['radar']).to(batch['radar'].device)
            sample_true = torch.from_numpy(
                samples['lidar']).to(batch['lidar'].device)

            sample_hat = sigmoid(self.model(radar))
            self.log_image_comparison("sample/val", sample_true, sample_hat)

        y_hat = self.model(batch['radar'])
        val_loss = self.loss(y_hat, batch['lidar'])
        val_chamfer = self.chamfer(y_hat > 0, batch['lidar'])

        self.log("loss/val", val_loss, sync_dist=True)
        self.log("chamfer/val", val_chamfer.to("cuda"), sync_dist=True)

    def evaluation_step(
        self, batch
    ) -> dict[str, Shaped[torch.Tensor, "batch"]]:
        """Evaluate model with mo metric aggregation."""
        y_hat = self.model(batch['radar'])
        y_hat_bool = (y_hat > 0)
        val_loss = self.loss(
            y_hat, batch['lidar'], reduce=False)
        val_chamfer = self.chamfer(y_hat_bool, batch['lidar'], reduce=False)
        val_hausdorff = self.hausdorff(
            y_hat_bool, batch['lidar'], reduce=False)
        return {
            "loss": val_loss, "chamfer": val_chamfer,
            "hausdorff": val_hausdorff}
