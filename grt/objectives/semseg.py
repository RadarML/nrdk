"""Radar training objectives and common building blocks for losses/metrics."""

from collections.abc import Mapping
from typing import Any, Generic, Protocol

import numpy as np
import torch
from abstract_dataloader.ext.objective import Objective, VisualizationConfig
from abstract_dataloader.ext.types import TArray
from einops import rearrange
from jaxtyping import Float, Integer, Shaped, UInt8
from torch import Tensor

from grt import vis


class SemsegData(Protocol, Generic[TArray]):
    """Protocol type for camera semantic segmentation data.

    Attributes:
        semseg: semantic segmentation data, with batch-height-width axis order.
    """

    semseg: UInt8[TArray, "batch h w"]


class Semseg(
    Objective[Tensor, SemsegData, Float[Tensor, "batch h w"]]
):
    """Semantic segmentation.

    Metrics:
        - `seg_loss`: categorical cross-entropy loss.
        - `seg_acc`: segmentation top-1 accuracy.
        - `seg_acc2`: segmentation top-2 accuracy.
        - `seg_miou`: segmentation Mean Intersection-Over-Union.

    Visualizations:
        - `semseg`: semantic segmentation class labels, colored according to
            the `semseg` color map provided to `vis_config.cmaps`.
            - Recommended: `tab10`, `tab20`, `jet`; `viridis` and `inferno` are
                okay.
            - Not recommended: colormaps with subtle variations in color.

    Args:
        weight: objective weight.
        vis_config: visualization configuration; the `cmaps` should have a
            `semseg` key.
    """

    def __init__(
        self, weight: float = 1.0,
        vis_config: VisualizationConfig | Mapping[str, Any] = {}
    ) -> None:
        self.weight = weight
        self.ce = torch.nn.CrossEntropyLoss(reduction='none')

        if not isinstance(vis_config, VisualizationConfig):
            vis_config = VisualizationConfig(**vis_config)
        self.vis_config = vis_config

    @staticmethod
    def __miou(
        y_true: Integer[Tensor, "batch h w"],
        y_hat: Integer[Tensor, "batch h w"], nc: int = 8
    ) -> Float[Tensor, "batch"]:
        y_true_onehot = torch.nn.functional.one_hot(y_true, num_classes=nc)
        y_hat_onehot = torch.nn.functional.one_hot(y_hat, num_classes=nc)
        intersection = torch.sum(y_true_onehot & y_hat_onehot, dim=(2, 3))
        union = torch.sum(y_true_onehot | y_hat_onehot, dim=(2, 3))
        return torch.mean(intersection / union, dim=1)

    def __call__(
        self, y_true: SemsegData, y_pred: Float[Tensor, "batch h w cls"],
        train: bool = True
    ) -> tuple[Float[Tensor, "batch"], dict[str, Float[Tensor, "batch"]]]:
        y_hat_logits = rearrange(
            y_pred, "b h w cls -> b cls h w")
        y_true_idx: Integer[Tensor, "b h w"] = (y_true.semseg.to(torch.long))

        loss = torch.mean(self.ce(y_hat_logits, y_true_idx), dim=(1, 2))

        with torch.no_grad():
            nc = y_hat_logits.shape[1]
            top2 = torch.topk(
                y_hat_logits, k=2, dim=1, largest=True, sorted=True).indices
            metrics = {
                "acc": torch.mean(
                    (top2[:, 0] == y_true_idx).to(torch.float32), dim=(1, 2)),
                "top2": torch.mean((
                    (top2[:, 0] == y_true_idx) | (top2[:, 1] == y_true_idx)
                ).to(torch.float32), dim=(1, 2)),
                "miou": self.__miou(y_true_idx, top2[:, 0], nc=nc),
            }
            metrics["bce"] = loss

        return loss, metrics

    def visualizations(
        self, y_true: SemsegData, y_pred: Float[Tensor, "batch h w cls"]
    ) -> dict[str, Shaped[np.ndarray, "H W 3"]]:
        y_hat_idx = torch.argmax(y_pred, dim=-1)
        semseg = vis.tile_images(
            y_true.semseg, y_hat_idx,
            cmap=self.vis_config.cmaps.get("semseg", "tab10"),
            cols=self.vis_config.cols)
        return {"semseg": semseg}

    def render(
        self, y_true: SemsegData, y_pred: Float[Tensor, "batch h w cls"],
        render_gt: bool = False
    ) -> dict[str, Shaped[np.ndarray, "H W 3"]]:
        y_hat_logits = rearrange(
            y_pred, "b el az rng cls -> b (rng cls) el az")
        y_hat_idx = torch.argmax(y_hat_logits, dim=1)

        res = {"semseg": y_hat_idx.to(torch.uint8).cpu().numpy()}
        if render_gt:
            res["semseg_gt"] = y_true.semseg.to(torch.uint8).cpu().numpy()

        return res
