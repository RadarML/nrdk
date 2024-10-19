"""Semantic labels."""

import numpy as np
import torch
from jaxtyping import Float, Shaped, UInt
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
    - `seg_miou`: segmentation Mean Intersection-Over-Union.
    """

    def __init__(self, weight: float = 1.0, cmap: str = 'tab10') -> None:
        self.weight = weight
        self.cmap = cmap
        self.ce = torch.nn.CrossEntropyLoss(reduction='none')

    def metrics(
        self, y_true: dict[str, Shaped[Tensor, "..."]],
        y_hat: dict[str, Shaped[Tensor, "..."]],
        reduce: bool = True, train: bool = True
    ) -> Metrics:
        """Get training metrics."""
        # Canonical elevation-azimuth-range -> elevation-azimuth
        seg_hat = y_hat["segment"][..., 0]

        target: UInt[Tensor, "batch h w"] = y_true["segment"].to(torch.long)
        input_logits: Float[Tensor, "batch c h w"] = seg_hat

        loss = torch.mean(self.ce(input_logits, target), dim=(1, 2))
        if reduce:
            loss = torch.mean(loss)

        with torch.no_grad():
            nc = input_logits.shape[1]
            input: UInt[Tensor, "batch h w"] = torch.argmax(input_logits, dim=1)

            acc = torch.mean((input == target).to(torch.float32), dim=(1, 2))
            target_onehot = torch.nn.functional.one_hot(target, num_classes=nc)
            input_onehot = torch.nn.functional.one_hot(input, num_classes=nc)
            intersection = torch.sum(target_onehot & input_onehot, dim=(2, 3))
            union = torch.sum(target_onehot | input_onehot, dim=(2, 3))
            miou = torch.mean(intersection / union, dim=1)

            metrics = {"seg_acc": acc, "seg_miou": miou}
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
        seg_hat = y_hat["segment"][..., 0]

        rez = Resize((180, 320), interpolation=InterpolationMode.NEAREST)
        y_hat_idx = torch.argmax(seg_hat, dim=1)
        return {"segment": comparison_grid(
            rez(y_true["segment"]), rez(y_hat_idx), cmap=self.cmap, cols=8)}
