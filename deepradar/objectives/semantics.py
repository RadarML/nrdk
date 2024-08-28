"""Semantic embeddings and labels."""

import numpy as np
import torch
from torch import Tensor
from torchvision.transforms import Resize, InterpolationMode
from jaxtyping import Shaped, Float
from beartype.typing import Optional

from deepradar.utils import comparison_grid
from .base import Metrics, Objective


class Embeddings(Objective):
    """Emulation/distillation of pre-computed spatial embeddings.

    Args:
        weight: objective weight.
    """

    def __init__(self, weight: float = 1.0) -> None:
        self.weight = weight

    def metrics(
        self, y_true: dict[str, Shaped[Tensor, "..."]],
        y_hat: dict[str, Shaped[Tensor, "..."]],
        reduce: bool = True, train: bool = True
    ) -> Metrics:
        """Get training metrics."""
        a = y_true["clip"]
        b = y_hat["clip"]

        norm = (torch.linalg.norm(a, dim=-1) * torch.linalg.norm(b, dim=-1))
        sim = torch.sum(torch.inner(a, b) / norm, dim=(1, 2))

        if reduce:
            sim = torch.mean(sim)

        return Metrics(loss=self.weight * sim, metrics={"seg_loss": sim})


class Segmentation(Objective):
    """Semantic segmentation.

    Args:
        weight: objective weight.
    """

    def __init__(self, weight: float = 1.0) -> None:
        self.weight = weight
        self.ce = torch.nn.CrossEntropyLoss(reduction='none')

    def metrics(
        self, y_true: dict[str, Shaped[Tensor, "..."]],
        y_hat: dict[str, Shaped[Tensor, "..."]],
        reduce: bool = True, train: bool = True
    ) -> Metrics:
        """Get training metrics."""
        a = y_true["segment"]
        b = y_hat["segment"]

        loss = torch.mean(self.ce(a, b), dim=(1, 2))
        acc = torch.mean(
            (torch.argmax(a, dim=1) == torch.argmax(b, dim=1)), dim=(1, 2))
        if reduce:
            loss = torch.mean(loss)
            acc = torch.mean(acc)

        return Metrics(
            loss=self.weight * loss,
            metrics={"seg_loss": loss, "seg_acc": acc})
