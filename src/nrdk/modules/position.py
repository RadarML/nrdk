"""Positional Embeddings."""

import torch
from beartype.typing import Optional, Sequence
from jaxtyping import Float
from torch import Tensor, nn


class Sinusoid(nn.Module):
    """Centered N-dimensional sinusoidal positional embedding.

    Channel dimensions are evenly divided between positional embeddings for
    each axis. Before applying the positional embedding, each positional index
    is also centered, scaled to `(-1, 1)`, and multiplied by a specified scale.

    !!! info

        We use the following equation for each axis:
        ```
        c = n_channels / n_spatial_dims / 2
        i = 1, 2, ... c
        w = coef ** (-i / c)
        t = scale * (j - d/2) / (d/2) = scale * (2j / d - 1)
        pos[2 * i] = sin(w * t)
        pos[2 * i + 1] = cos(w * t)
        ```
        where `d` is the dimension of the axis, and `j` is the index in that
        axis.

    Args:
        scale: position embedding scale (i.e. the spatial range that this axis
            corresponds to). If `None`, only the global scale is used.
        global_scale: scalar constant to multiply scale by for convenience of
            representation; yields a net scale of `scale * global_scale`.
        coef: geometric frequency progression base coefficient; the
            frequencies are given by `coef ** (-i / c)`.
    """

    def __init__(
        self, scale: Optional[Sequence[float]] = None,
        global_scale: float = 1.0, coef: float = 10000.0
    ) -> None:
        super().__init__()
        if scale is None:
            self.scale = [global_scale]
        else:
            self.scale = [s * global_scale for s in scale]
        self.coef = coef

    def forward(
        self, x: Float[Tensor, "batch *spatial channels"]
    ) -> Float[Tensor, "batch *spatial channels"]:
        """Apply sinusoidal embedding.

        Data should be in batch-spatial-feature order.
        """
        # w = coef ** (-i / c)
        nd = len(x.shape) - 2
        c = x.shape[-1] // 2 // nd
        i = torch.arange(c, device=x.device)
        w = self.coef ** (-i / c)

        start_dim = 0
        for axis, (d, scale) in enumerate(zip(x.shape[1:-1], self.scale * nd)):
            # t = scale * (j - d/2) / (d/2) = scale * (2j / d - 1)
            t = scale * (2 * (torch.arange(d, device=x.device) - 0.5) / d - 1)
            wt: Float[Tensor, "d c"] = t[:, None] * w[None, :]

            p_slice = [None] * (len(x.shape) - 1) + [slice(None)]
            p_slice[axis + 1] = slice(None)

            # pos[2 * i] = sin(w * t)
            x_sin_slice = [slice(None)] * len(x.shape)
            x_sin_slice[-1] = slice(start_dim, start_dim + c * 2, 2)
            x[x_sin_slice] = x[x_sin_slice] + torch.sin(wt)[p_slice]

            # pos[2 * i + 1] = cos(w * t)
            x_cos_slice = [slice(None)] * len(x.shape)
            x_cos_slice[-1] = slice(start_dim + 1, start_dim + c * 2 + 1, 2)
            x[x_cos_slice] = x[x_cos_slice] + torch.cos(wt)[p_slice]

            start_dim += c * 2

        return x


class LearnableND(nn.Module):
    """Learnable N-dimensional positional embedding.

    Args:
        d_model: Model feature dimensions.
        shape: Shape of input positions; must be fixed.
    """

    def __init__(
        self, d_model: int = 512,
        shape: list[int] | tuple[int, ...] = (16, 16)
    ) -> None:
        super().__init__()
        self.shape = shape
        self.embeddings = nn.ParameterList([
            torch.normal(0, 0.02, (d, d_model)) for d in shape])

    def forward(self, x: Float[Tensor, "n ... c"]) -> Float[Tensor, "n ... c"]:
        """Apply embeddings.

        Data should be in batch-spatial-feature order.
        """
        for i, e in enumerate(self.embeddings):
            idxs = [None] * (len(self.shape) + 1) + [slice(None)]
            idxs[i + 1] = slice(None)
            x = x + e[idxs]
        return x


class Readout(nn.Module):
    """Add readout token (contatenating along the spatial axis).

    !!! info

        The readout token is always added as the last token in the sequence.

    Args:
        d_model: model feature dimensions.
    """

    def __init__(self, d_model: int = 512) -> None:
        super().__init__()
        self.readout = nn.Parameter(data=torch.normal(0, 0.02, (d_model,)))

    def forward(self, x: Float[Tensor, "n s c"]) -> Float[Tensor, "n s2 c"]:
        """Concatenate readout token."""
        readout = torch.tile(self.readout[None, None, :], (x.shape[0], 1, 1))
        return torch.concatenate((x, readout), dim=1)


class BasisChange(nn.Module):
    """Create "change-of-basis" query.

    Uses a 'reference vector', e.g. the output for a readout token or the
    token-wise mean of the output. The vector is tiled, and a sinusoidal
    embedding ([`Sinusoid`][^.]) is applied.

    Args:
        shape: query shape.
        flatten: whether to flatten spatial axes (e.g. for spatial-agnostic
            decoders such as generic transformers).
        scale: position embedding scale (i.e. the spatial range that this axis
            corresponds to). If `None`, only the global scale is used.
        global_scale: scalar constant to multiply scale by for convenience of
            representation; yields a net scale of `scale * global_scale`.
    """

    def __init__(
        self, shape: Sequence[int] = [], flatten: bool = True,
        scale: Optional[Sequence[float]] = None, global_scale: float = 1.0
    ) -> None:
        super().__init__()

        self.pos = Sinusoid(scale=scale, global_scale=global_scale)
        self.shape = shape
        self.flatten = flatten

    def forward(self, x: Float[Tensor, "n c"]) -> Float[Tensor, "n *t2 c"]:
        """Apply change of basis.

        Args:
            x: input reference vector. Should only be a single vector per
                batch entry.

        Returns:
            Input reference `x`, expanded to the query shape, with a sinusoidal
                positional embedding added. The tensor is flattened along the
                spatial axes if desired (`flatten=True`).
        """
        idxs = [slice(None)] + [None] * len(self.shape) + [slice(None)]
        query = self.pos(
            torch.tile(x[idxs], (1, *self.shape, 1)))

        if self.flatten:
            query = query.reshape(x.shape[0], -1, x.shape[-1])
        return query
