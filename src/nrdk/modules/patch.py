"""Patching and unpatching modules."""

from collections.abc import Sequence

import numpy as np
import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn


class PatchMerge(nn.Module):
    """Merge patches with normalization and nominally reduced projection.

    !!! tip "Supports an arbitrary number of spatial dimensions."

    Args:
        d_in: embedding dimension.
        d_out: output dimension; nominally less than `d_in * prod(scale)`.
        scale: downsampling factor to apply in each axis.
        norm: whether to perform layer normalization.
    """

    def __init__(
        self, d_in: int, d_out: int, scale: Sequence[int] = [],
        norm: bool = True
    ) -> None:
        super().__init__()

        self.scale = scale
        d_merge = d_in * int(np.prod(scale))
        self.linear = nn.Linear(d_merge, d_out, bias=False)
        self.norm = nn.LayerNorm(d_merge) if norm else None

    def _merge(self, x: Float[Tensor, "n *t c"]) -> Float[Tensor, "n *t2 c2"]:
        """Perform patch merging."""
        n, *t, c = x.shape
        # n d1//s1 s1 d2//s2 s2 ... dN//sN sN c
        dims = sum(([d // s, s] for d, s in zip(t, self.scale)), start=[n])
        # new order: 0, [1, 3, 5, ...], -1, [2, 4, 6, ...]
        order = (
            [0] + [2 * i + 1 for i in range(len(self.scale))]
            + [2 * i + 2 for i in range(len(self.scale))] + [-1])
        # new shape: n d1//s1 d2//s2 ... dN//sN (s1 s2 ... sN c)
        t2 = [d // s for d, s in zip(t, self.scale)]
        return x.reshape(dims + [c]).permute(order).reshape(n, *t2, -1)

    def forward(self, x: Float[Tensor, "n *t c"]) -> Float[Tensor, "n *t2 c2"]:
        """Merge and project."""
        merged = self._merge(x)
        if self.norm is not None:
            merged = self.norm(merged)
        return self.linear(merged)


class Unpatch(nn.Module):
    """Unpatch data.

    !!! bug "Todo"

        This module is manually implemented for each `n`, and is currently only
        implemented for `n=2,3,4` spatial dimensions.

    Args:
        output_size: output n-dimensional shape.
        features: number of input features; should be `>= size * size`.
        size: patch size as (width, height, channels).
    """

    def __init__(
        self, output_size: Sequence[int],
        features: int = 512, size: Sequence[int] = (16, 16)
    ) -> None:
        super().__init__()

        self.linear = nn.Linear(features, output_size[-1] * int(np.prod(size)))
        self.size = size
        self.output_size = output_size

    def forward(
        self, x: Float[Tensor, "n xin c"]
    ) -> Float[Tensor, "n *xout c_out"]:
        """Perform n-dimensional unpatching.

        Operates in batch-spatial-feature order; spatial axes are flattened on
            the input, and unflattened in the output.
        """
        embedding = self.linear(x)

        if len(self.size) == 2:
            return rearrange(
                embedding, "n (x1 x2) (s1 s2 c) -> n (x1 s1) (x2 s2) c",
                x1=self.output_size[0] // self.size[0],
                x2=self.output_size[1] // self.size[1],
                s1=self.size[0], s2=self.size[1], c=self.output_size[-1])
        elif len(self.size) == 3:
            return rearrange(
                embedding,
                "n (x1 x2 x3) (s1 s2 s3 c) -> n (x1 s1) (x2 s2) (x3 s3) c",
                x1=self.output_size[0] // self.size[0],
                x2=self.output_size[1] // self.size[1],
                x3=self.output_size[2] // self.size[2],
                s1=self.size[0], s2=self.size[1], s3=self.size[2],
                c=self.output_size[-1])
        elif len(self.size) == 4:
            return rearrange(
                embedding,
                "n (x1 x2 x3 x4) (s1 s2 s3 s4 c) -> "
                "n (x1 s1) (x2 s2) (x3 s3) (x4 s4) c",
                x1=self.output_size[0] // self.size[0],
                x2=self.output_size[1] // self.size[1],
                x3=self.output_size[2] // self.size[2],
                x4=self.output_size[3] // self.size[3],
                s1=self.size[0], s2=self.size[1],
                s3=self.size[2], s4=self.size[3],
                c=self.output_size[-1])
        else:
            raise ValueError(
                "Unpatch is only implemented for 2/3/4D tensors.")


class Squeeze(nn.Module):
    """Remove axes by moving them to the channel axis.

    Args:
        dim: axes to remove, indexed with respect to the spatial dimensions
            (so the first spatial axis is `0`, and so on).
        size: optional original spatial axis sizes; used to compute the number
            of resulting channels.
    """

    def __init__(
        self, dim: Sequence[int] = [],
        size: Sequence[int] | None = None
    ) -> None:
        super().__init__()
        self.dim = sorted(dim, reverse=True)
        self.size = size

    @property
    def n_channels(self) -> int:
        """Get number of channels."""
        if self.size is None:
            raise ValueError(
                "Cannot compute n_channels without a provided `size`.")
        nc = 1
        for d in self.dim:
            nc *= self.size[d]
        return nc

    def forward(
        self, data: Float[Tensor, "batch *spatial channels"]
    ) -> Float[Tensor, "batch *spatial2 channels2"]:
        """Apply transform."""
        for dim in self.dim:
            data = torch.moveaxis(data, dim + 1, -1)
            data = data.reshape(*data.shape[:-2], -1)
        return data
