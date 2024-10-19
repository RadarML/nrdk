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

    Uses the following equation for each axis::

        c = n_channels / n_spatial_dims / 2
        i = 1, 2, ... c
        w = coef ** (-i / c)
        t = scale * (j - d/2) / (d/2) = scale * (2j / d - 1)
        pos[2 * i] = sin(w * t)
        pos[2 * i + 1] = cos(w * t)

    where `d` is the dimension of the axis, and `j` is the index in that axis.

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


class Rotary2D(nn.Module):
    """2D rotary encoding [M2]_.

    Args:
        features: number of features in this embedding.
    """

    def __init__(self, features: int = 512) -> None:
        super().__init__()
        self.d = features // 2

    def _get_R(self, m, theta) -> Float[torch.Tensor, "x d 2 2"]:
        """Closure on theta."""
        mt = m[:, None] * theta[None, :]
        cos = torch.cos(mt)
        sin = torch.sin(mt)
        R_stack = torch.stack(
            [torch.stack([cos, -sin]), torch.stack([sin, cos])])
        return torch.moveaxis(R_stack, (0, 1, 2, 3), (2, 3, 0, 1))

    def forward(
        self, x: Float[Tensor, "b x1 x2 f"]
    ) -> Float[Tensor, "b x1 x2 f"]:
        """Apply 2D rotary encoding.

        Data should be in batch-spatial-feature order.
        """
        i = torch.arange(x.shape[3] // 4, device=x.device)
        theta = 100000 ** (-i / x.shape[1])
        m1 = torch.arange(x.shape[1], device=x.device)
        m2 = torch.arange(x.shape[2], device=x.device)

        x1: Float[torch.Tensor, "b x1 x2 d 2 1"] = (
            x[..., :self.d].reshape(*x.shape[:-1], -1, 2)[..., None])
        R1: Float[torch.Tensor, "1 x1  1 d 2 2"] = (
            self._get_R(m1, theta)[None, :, None, :, :, :])
        x1_emb = torch.matmul(R1, x1).reshape(*x.shape[:-1], -1)

        x2: Float[torch.Tensor, "b x1 x2 d 2 1"] = (
            x[..., self.d:].reshape(*x.shape[:-1], -1, 2)[..., None])
        R2: Float[torch.Tensor, "1 1  x2 d 2 2"] = (
            self._get_R(m2, theta)[None, None, :, :, :, :])
        x2_emb = torch.matmul(R2, x2).reshape(*x.shape[:-1], -1)

        return torch.concatenate([x1_emb, x2_emb], dim=3)


class Learnable1D(nn.Module):
    """Learnable 1-dimensional positional embedding.

    Args:
        d_model: Model feature dimension.
        size: Number of positions; must be fixed (i.e. operates on fixed
            dimension input only).
    """

    def __init__(self, d_model: int = 512, size: int = 1024) -> None:
        super().__init__()
        self.embeddings = nn.Parameter(
            data=torch.normal(0, 0.02, (size, d_model)))

    def forward(self, x: Float[Tensor, "n #t c"]) -> Float[Tensor, "n t c"]:
        """Apply positional embedding.

        Data should be in batch-spatial-feature order.
        """
        return x + self.embeddings


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
