"""Classification metrics."""

from typing import Literal

import einops
import torch
from jaxtyping import Bool, Float
from torch import Tensor
from torch.nn.functional import binary_cross_entropy_with_logits


class BCE:
    """Weighted Binary Cross Entropy (BCE) loss for polar occupancy.

    Args:
        positive_weight: additional weight to assign to occupied cells; all
            unoccupied cells are always given `0.0` weight.
        weighting: apply equal area weighting. If `cylindrical`, assumes a
            cylindrical coordinate system, and multiplies by range; if
            `spherical`, assumes a spherical coordinate system, and multiplies
            by `range**2`.
    """

    def __init__(
        self, positive_weight: float = 1.0,
        weighting: Literal["cylindrical", "spherical"] | None = None
    ) -> None:
        self.weighting = weighting
        self.positive_weight = positive_weight

    def __call__(
        self, y_true: Bool[Tensor, "batch y z range"],
        y_hat: Float[Tensor, "batch y z range"]
    ) -> Float[Tensor, "batch"]:
        """Compute BCE loss.

        Args:
            y_true: true 3D occupany grid (`True` if occupied). If weighting is
                enabled, the range axis should be the first spatial axis.
            y_hat: predicted occupancy logits.

        Returns:
            BCE loss for each item in the batch.
        """
        *_, nr = y_true.shape

        with torch.device(y_true.device):
            if self.weighting == "cylindrical":
                weight = ((torch.arange(nr) + 1))[..., :]
            elif self.weighting == "spherical":
                weight = ((torch.arange(nr) + 1))[..., :] ** 2
            else:
                weight = torch.ones((1, 1, 1, 1), dtype=y_hat.dtype)

        if self.positive_weight != 1.0:
            weight = torch.where(y_true, self.positive_weight, 1.0) * weight

        weight = weight.expand(y_true.shape)
        total_weight = einops.reduce(weight, "batch y z range -> batch", "sum")
        loss = einops.reduce(
            weight * binary_cross_entropy_with_logits(
                y_hat, y_true.to(y_hat.dtype), reduction='none'),
            "batch y z range -> batch", "sum")

        return loss / total_weight


class BinaryDiceLoss:
    """Binary Dice loss for classification."""

    def __init__(
        self, weighting: Literal["cylindrical", "spherical"] | None = None
    ) -> None:
        self.weighting = weighting

    def __call__(
        self, y_true: Bool[Tensor, "batch y z range"],
        y_hat: Float[Tensor, "batch y z range"]
    ) -> Float[Tensor, "batch"]:
        """Compute dice loss for binary classification."""
        *_, nr = y_true.shape

        with torch.device(y_true.device):
            if self.weighting == "cylindrical":
                weight = ((torch.arange(nr) + 1))[..., :]
            elif self.weighting == "spherical":
                weight = ((torch.arange(nr) + 1))[..., :] ** 2
            else:
                weight = torch.ones((1, 1, 1, 1), dtype=y_hat.dtype)

        denominator = (
            torch.sum(y_hat * y_hat * weight, dim=(1, 2))
            + torch.sum(y_true * weight, dim=(1, 2)))
        numerator = 2 * torch.sum(y_hat * y_true * weight, dim=(1, 2))
        return 1.0 - numerator / denominator


class FocalLoss:
    """Focal loss for classification.

    Analogous to `torch.binary_cross_entropy_with_logits`; see ["Focal Loss for
    Dense Object Detection"](https://arxiv.org/abs/1708.02002v2).

    Args:
        gamma: focusing parameter; the original paper suggests
            `gamma = 2.0` as a good default.
        eps: epsilon for clipping the focusing coefficient `(1 - p_t)**gamma`
            to avoid `NaN` gradients caused by floating point errors when
            this is close to 0 or 1. By default, we select `1e-4` to be
            larger than the maximum normal for IEEE float16 (`~6.1 x 10^-5`).
    """

    def __init__(self, gamma: float = 2.0, eps: float = 1e-4) -> None:
        self.gamma = gamma
        self.eps = eps

    def __call__(
        self, y_true: Bool[Tensor, "batch *spatial"],
        y_hat: Float[Tensor, "batch *spatial"]
    ) -> Float[Tensor, "batch"]:
        """Compute focal loss.

        Args:
            y_true: true labels.
            y_hat: predicted logits.

        Returns:
            Focal loss for each entry in the batch.
        """
        # d = 1 + exp(y_hat)
        log_d = torch.logaddexp(y_hat, torch.tensor([0.], device=y_hat.device))
        d = torch.exp(y_hat) + 1.0

        # y_t = exp(y_hat)          y_true = True
        #       1                   otherwise
        log_y_t = torch.where(y_true, y_hat, 0.0)
        y_t = torch.exp(log_y_t)

        # p_t = sigmoid(y_hat)      y_true = True
        #       1 - sigmoid(y_hat)  otherwise
        p_t = y_t / d
        log_p_t = log_y_t - log_d

        # FL = - (1 - p_t)**gamma * log(p_t)
        focus = torch.clip((1 - p_t) ** self.gamma, self.eps, 1 - self.eps)
        fl = - focus * log_p_t

        return fl
