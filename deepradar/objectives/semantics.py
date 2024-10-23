"""Semantic labels."""

import numpy as np
import torch
from beartype.typing import Any
from einops import rearrange
from jaxtyping import Float, Integer, Shaped
from torch import Tensor
from torchvision.transforms import InterpolationMode, Resize

from deepradar.utils import comparison_grid

from .base import Metrics, Objective


class Segmentation(Objective):
    """Semantic segmentation.

    Args:
        weight: objective weight.
        cmap: colors to use for visualizations. Recommended: `tab10`,
            `tab20`, `jet`. `viridis` and `inferno` are okay. Not recommended:
            any colormaps with only subtle variations in color.

    Metrics:

    - `seg_acc`: segmentation top-1 accuracy.
    - `seg_acc2`: segmentation top-2 accuracy.
    - `seg_miou`: segmentation Mean Intersection-Over-Union.
    """

    def __init__(self, weight: float = 1.0, cmap: str = 'tab10') -> None:
        self.weight = weight
        self.cmap = cmap
        self.ce = torch.nn.CrossEntropyLoss(reduction='none')

    @staticmethod
    def miou(
        y_true: Integer[Tensor, "batch h w"],
        y_hat: Integer[Tensor, "batch h w"], nc: int = 8
    ) -> Float[Tensor, "batch"]:
        """Compute Intersection over Union for the given class labels."""
        y_true_onehot = torch.nn.functional.one_hot(y_true, num_classes=nc)
        y_hat_onehot = torch.nn.functional.one_hot(y_hat, num_classes=nc)
        intersection = torch.sum(y_true_onehot & y_hat_onehot, dim=(2, 3))
        union = torch.sum(y_true_onehot | y_hat_onehot, dim=(2, 3))
        return torch.mean(intersection / union, dim=1)

    def metrics(
        self, y_true: dict[str, Shaped[Tensor, "..."]],
        y_hat: dict[str, Shaped[Tensor, "..."]],
        reduce: bool = True, train: bool = True
    ) -> Metrics:
        """Get training metrics."""
        # The `rng` axis should be a singleton.
        y_hat_logits = rearrange(
            y_hat["segment"], "b el az rng cls -> b (rng cls) el az")

        y_true_idx: Integer[Tensor, "b h w"] = y_true["segment"].to(torch.long)

        loss = torch.mean(self.ce(y_hat_logits, y_true_idx), dim=(1, 2))
        if reduce:
            loss = torch.mean(loss)

        with torch.no_grad():
            nc = y_hat_logits.shape[1]
            top2 = torch.topk(
                y_hat_logits, k=2, dim=1, largest=True, sorted=True).indices
            metrics = {
                "seg_acc": torch.mean(
                    (top2[:, 0] == y_true_idx).to(torch.float32), dim=(1, 2)),
                "seg_top2": torch.mean((
                    (top2[:, 0] == y_true_idx) | (top2[:, 1] == y_true_idx)
                ).to(torch.float32), dim=(1, 2)),
                "seg_moiu": self.miou(y_true_idx, top2[:, 0], nc=nc)
            }

            if reduce:
                metrics = {k: torch.mean(v) for k, v in metrics.items()}
            metrics["seg_loss"] = loss

        return Metrics(loss=self.weight * loss, metrics=metrics)

    def visualizations(
        self, y_true: dict[str, Shaped[Tensor, "..."]],
        y_hat: dict[str, Shaped[Tensor, "..."]]
    ) -> dict[str, Shaped[np.ndarray, "H W 3"]]:
        """Generate visualizations."""
        # Canonical elevation-azimuth-range -> elevation-azimuth
        y_hat_logits = rearrange(
            y_hat["segment"], "b el az rng cls -> b (rng cls) el az")
        y_hat_idx = torch.argmax(y_hat_logits, dim=1)

        rez = Resize((180, 320), interpolation=InterpolationMode.NEAREST)
        return {"seg": comparison_grid(
            rez(y_true["segment"]), rez(y_hat_idx), cmap=self.cmap, cols=8)}

    RENDER_CHANNELS: dict[str, dict[str, Any]] = {
        "seg": {
            "format": "lzma", "type": "u1", "shape": [160, 160],
            "desc": "Predicted segmentation maps."},
        "seg_gt": {
            "format": "lzma", "type": "u1", "shape": [160, 160],
            "desc": "Ground truth segmentation maps."}
    }

    def render(
        self, y_true: dict[str, Shaped[Tensor, "batch ..."]],
        y_hat: dict[str, Shaped[Tensor, "batch ..."]]
    ) -> dict[str, Shaped[np.ndarray, "batch ..."]]:
        """Summarize predictions to visualize later.

        Args:
            y_true, y_hat: see :py:meth:`Objective.metrics`.

        Returns:
            A dict, where each key is the name of a visualization or output
            data, and the value is a quantized or packed format if possible.
        """
        y_hat_logits = rearrange(
            y_hat["segment"], "b el az rng cls -> b (rng cls) el az")
        y_hat_idx = torch.argmax(y_hat_logits, dim=1)
        return {
            "seg": y_hat_idx.to(torch.uint8).cpu().numpy(),
            "seg_gt": y_true["segment"].to(torch.uint8).cpu().numpy()
        }
