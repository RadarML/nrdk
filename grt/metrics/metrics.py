"""Train / Test Metrics."""

from typing import TypeVar, cast

import torch
from jaxtyping import Bool, Float
from torch import Tensor

T = TypeVar("T", bound=Tensor)


def lp_power(x: T, ord: int | float = 1) -> T:
    """Compute lp loss power `|x|^p`.

    The caller is responsible for forming the actual lp loss.

    !!! note

        This implementation provides a few optimized special cases:
        - `ord = 0`: directly compute `x != 0`.
        - `ord = 1`: use `abs`.
        - `ord = 2`: use `square`.

    Args:
        x: the `y_true - y_hat` difference.
        ord: loss order, i.e. the `p` in Lp.

    Returns:
        Element-wise `|x|^p`, with optimized cases for `p=0,1,2`.
    """
    if ord == 0:
        return cast(T, (x != 0).to(x.dtype))
    elif ord == 1:
        return cast(T, torch.abs(x))
    elif ord == 2:
        return cast(T, torch.square(x))
    else:
        return cast(T, torch.abs(x) ** ord)


def mean_with_mask(
    x: Float[Tensor, "batch ..."], mask: Bool[Tensor, "batch ..."] | None
) -> Float[Tensor, "batch"]:
    """Take the mean of a tensor across non-batch axes.

    Args:
        x: input values, with a leading batch axis.
        mask: optional validity mask to apply.

    Returns:
        Mean of `x` across non-batch axes, where only values which are
            `True` in the mask (if provided) contribute to the mean.
    """
    batch = x.shape[0]
    if mask is not None:
        masked = mask * x
        sample_losses = torch.sum(masked.reshape(batch, -1), dim=1)
        n_valid = torch.clamp(
            torch.sum(mask.reshape(batch, -1), dim=-1), min=1)
        return sample_losses / n_valid
    else:
        return torch.mean(x.reshape(batch, -1), dim=1)


class Lp:
    """Generic Lp loss.

    Args:
        ord: loss order (the `p` in Lp).
    """

    def __init__(self, ord: int | float = 1) -> None:
        self.ord = ord

    def __call__(
        self, y_true: Float[Tensor, "batch ..."],
        y_hat: Float[Tensor, "batch ..."],
        valid: Bool[Tensor, "batch *spatial"] | None = None
    ) -> Float[Tensor, "batch"]:
        """Compute Lp loss."""
        diff = lp_power(y_true - y_hat, ord=self.ord)
        return mean_with_mask(diff, valid)


class VoxelDepth:
    """Depth error for voxel grids, as measured by the first occupied voxel.

    Returns the error in grid units.

    !!! warning "Not Differentiable"

    Args:
        axis: spatial axis to measure depth along, indexed not counting the
            batch axis (so `0` is the first spatial axis).
        reverse: measure depth "reversed", i.e. first voxel from highest index
            first instead of first voxel from the lowest index first.
        ord: lp loss order; any non-negative real is supported (i.e. all
            lp losses, excluding infinity).
        filter_empty: Whether to disregard empty bins from the calculation, as
            measured by the ground truth.
    """

    def __init__(
        self, axis: int = 0, reverse: bool = False, ord: float | int = 1,
        filter_empty: bool = True
    ) -> None:
        self.axis = axis
        self.reversed = reverse
        self.ord = ord
        self.filter_empty = filter_empty

    def _lp_loss(
        self, y_true: Float[Tensor, "batch *spatial"],
        y_hat: Float[Tensor, "batch *spatial"],
        mask: Bool[Tensor, "batch *spatial"] | None = None
    ) -> Float[Tensor, "batch"]:
        """Compute lp depth loss.

        Args:
            y_true: true 3D occupany grid (`True` if occupied). If weighting is
                enabled, the range axis should be the first spatial axis.
            y_hat: predicted occupancy logits.
            mask: optional mask of valid depth points, e.g. to exclude lidar
                beams with no return.

        Returns:
            Difference in depth, as measured by the first occupied grid cell
                in each azimuth/elevation bin.
        """
        diff = lp_power(y_true - y_hat)
        batch = y_true.shape[0]
        if mask is None:
            return torch.mean(diff.reshape(batch, -1), dim=1)
        else:
            total = torch.sum((diff * mask).reshape(batch, -1), dim=1)
            n_valid = torch.sum(mask.reshape(batch, -1), dim=1)
            return total / n_valid

    def __call__(
        self, y_true: Bool[Tensor, "batch *spatial"],
        y_hat: Bool[Tensor, "batch *spatial"]
    ) -> Float[Tensor, "batch"]:
        """Compute depth loss."""

        def _to_depth(y) -> Float[Tensor, "batch *spatial_1"]:
            if self.reversed:
                y = torch.flip(y, dims=(self.axis + 1,))
            # Need to cast to uint8 since argmax is not implemented on bool
            # This should be a pure cast with no actual transform
            return torch.argmax(
                y.to(torch.uint8), dim=self.axis + 1).to(torch.float)

        if self.filter_empty:
            mask = torch.any(y_true, dim=self.axis + 1)
        else:
            mask = None
        return self._lp_loss(_to_depth(y_true), _to_depth(y_hat), mask=mask)


class DepthWithConfidence:
    """Depth loss with confidence.

    Uses the formulation given by [dust3r](https://github.com/naver/dust3r):
    ```
    L = C * |y_true - y_hat| - a * log(C)
    ```

    Args:
        alpha: confidence scaling parameter.
        ord: lp loss order. Note that `ord=1` actually corresponds to a
            geometric `l2` loss, i.e. the distance between true and predicted
            points in 3D space.
    """

    def __init__(self, alpha: float = 0.2, ord: float | int = 1) -> None:
        self.ord = ord
        self.alpha = alpha

    def __call__(
        self, y_true: Float[Tensor, "batch *spatial"],
        y_hat: Float[Tensor, "batch *spatial"],
        confidence: Float[Tensor, "batch *spatial"],
        valid: Bool[Tensor, "batch *spatial"] | None = None
    ) -> Float[Tensor, "batch"]:
        """Compute lp depth loss.

        Args:
            y_true: actual depth.
            y_hat: predicted depth.
            confidence: predicted confidence values for each azimuth-elevation
                depth bin.
            valid: mask of valid depth points, e.g. to exclude lidar beams with
                no return.

        Returns:
            Confidence-aware depth loss for each entry in the batch.
        """
        diff = lp_power(y_true - y_hat)
        c = 1 + torch.exp(confidence)

        pixel_losses = (c * diff - self.alpha - torch.log(c))
        return mean_with_mask(pixel_losses, valid)
