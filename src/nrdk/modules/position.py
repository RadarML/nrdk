"""Positional Embeddings."""

from collections.abc import Sequence
from typing import cast

import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor, nn


class Sinusoid(nn.Module):
    """Centered N-dimensional sinusoidal positional embedding.

    Channel dimensions are evenly divided between positional embeddings for
    each axis. Before applying the positional embedding, each positional index
    is also centered, scaled to `(-1, 1)`, and multiplied by a specified scale.

    Alternatively, if a `positions` argument is provided for each axis, these
    values are used instead of the positional indices; note that provided
    positions are also multiplied by the scale.

    Args:
        scale: position embedding scale (i.e. the spatial range that this axis
            corresponds to). If `None`, uses `scale=1.0`.
        w_min: minimum wavelength for each axis, specified in the same units as
            the scaled positions for each axis. If `None`, uses `w_min=1.0`.
        coef: geometric frequency progression base coefficient; the
            frequencies are given by `coef ** (-i / c)`.
        channels: number or proportion of channels to use for each axis. If
            `None`, the input channels are split equally (rounded down,
            remainder unused) between each axis.
    """

    def __init__(
        self, scale: Sequence[float] | float | None = None,
        w_min: Sequence[float] | float | None = None, coef: float = 10000.0,
        channels: Sequence[int] | Sequence[float] | None = None
    ) -> None:
        super().__init__()
        self.scale = scale
        self.w_min = w_min
        self.coef = coef
        self.channels = channels

    def _get_channels(self, shape: Sequence[int]) -> Sequence[int]:
        """Get positional encoding channel size allocation for each axis."""
        nd = len(shape) - 2
        if self.channels is None:
            return [shape[-1] // 2 // nd] * nd
        else:
            channels = self.channels
            if len(channels) != nd:
                raise ValueError(
                    f"Channels {channels} specify {len(channels)} "
                    f"dimensions, but the input has {nd} dimensions.")

            if any(isinstance(c, float) for c in channels):
                channels = [int(c * shape[-1] // 2) for c in channels]
            channels = cast(Sequence[int], channels)

            if sum(channels) * 2 > shape[-1]:
                raise ValueError(
                    f"Not enough channels: positional encodings [sin, cos] x "
                    f"{nd} dimensions require sum({channels}) * 2 = "
                    f"{sum(channels) * 2} channels, but the input has "
                    f"{shape[-1]} channels.")

            return channels

    def _get_scale(self, shape: Sequence[int]) -> Sequence[float]:
        """Get positional encoding scale for each axis."""
        nd = len(shape) - 2
        if self.scale is None:
            scale = [1.0] * nd
        elif isinstance(self.scale, float):
            scale = [self.scale] * nd
        else:
            # Somehow pyright believes self.scale can be an int now?
            assert isinstance(self.scale, Sequence)
            scale = self.scale
            if len(scale) != nd:
                raise ValueError(
                    f"Scale {scale} specifies {len(scale)} dimensions, but "
                    f"the input has {nd} dimensions.")

        if self.w_min is None:
            w_min = [1.0] * nd
        elif isinstance(self.w_min, float):
            w_min = [self.w_min] * nd
        else:
            assert isinstance(self.w_min, Sequence)
            w_min = self.w_min
            if len(w_min) != nd:
                raise ValueError(
                    f"Minimum wavelength {w_min} specifies "
                    f"{len(w_min)} dimensions, but the input has {nd} "
                    f"dimensions.")

        return [s * np.pi / w for s, w in zip(scale, w_min)]

    def forward(
        self, x: Float[Tensor, "batch *spatial channels"],
        positions: Sequence[Float[Tensor, "#batch _length"]] | None = None
    ) -> Float[Tensor, "batch *spatial channels"]:
        """Apply sinusoidal embedding.

        Args:
            x: input data.
            positions: optional exact position values to use for each axis. If
                `None`, uses the positional indices, scaled to `[-1, 1]`.

        Returns:
            Input with a sinusoidal positional encoding added.
        """
        if positions is None:
            positions = [
                torch.linspace(-1.0, 1.0, steps=n, device=x.device)[None, :]
                for n in x.shape[1:-1]]

        if len(positions) != len(x.shape) - 2:
            raise ValueError(
                f"Received {len(positions)} positions, but the input with "
                f"shape {x.shape} has {len(x.shape) - 2} spatial axes.")

        channels = self._get_channels(x.shape)
        scales = self._get_scale(x.shape)

        start_dim = 0
        for axis, (t, sc, nc) in enumerate(zip(positions, scales, channels)):
            w = self.coef ** (-torch.arange(nc, device=x.device) / nc)
            wt = sc * t[:, :, None] * w[None, None, :]

            # Apply along the target spatial axis, batch, channels
            # Broadcast on all others
            # p_slice = [:, None, ..., None, :, None, ..., None, :]
            p_slice: list[slice | None] = [None] * len(x.shape)
            p_slice[0] = slice(None)
            p_slice[axis + 1] = slice(None)
            p_slice[-1] = slice(None)

            # pos[2 * i] = sin(w * t)
            x_sin_slice = [slice(None)] * len(x.shape)
            x_sin_slice[-1] = slice(start_dim, start_dim + nc * 2, 2)
            x[x_sin_slice] = x[x_sin_slice] + torch.sin(wt)[p_slice]

            # pos[2 * i + 1] = cos(w * t)
            x_cos_slice = [slice(None)] * len(x.shape)
            x_cos_slice[-1] = slice(start_dim + 1, start_dim + nc * 2 + 1, 2)
            x[x_cos_slice] = x[x_cos_slice] + torch.cos(wt)[p_slice]

            start_dim += nc * 2

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
        scale: position embedding scale.
        w_min: minimum wavelength for each axis.
        coef: geometric frequency coefficient / scale.
    """

    def __init__(
        self, shape: Sequence[int] = [], flatten: bool = True,
        scale: Sequence[float] | float | None = None,
        w_min: Sequence[float] | float | None = None,
        coef: float = 10000.0,
    ) -> None:
        super().__init__()

        self.pos = Sinusoid(scale=scale, w_min=w_min, coef=coef)
        self.shape = shape
        self.flatten = flatten

    def forward(
        self, x: Float[Tensor, "n c"],
        positions: Sequence[Float[Tensor, "#batch _length"]] | None = None
    ) -> Float[Tensor, "n *t2 c"]:
        """Apply change of basis.

        Args:
            x: input reference vector. Should only be a single vector per
                batch entry.
            positions: exact position values for each axis.

        Returns:
            Input reference `x`, expanded to the query shape, with a sinusoidal
                positional embedding added. The tensor is flattened along the
                spatial axes if desired (`flatten=True`).
        """
        idxs = [slice(None)] + [None] * len(self.shape) + [slice(None)]
        query = self.pos(
            torch.tile(x[idxs], (1, *self.shape, 1)),
            positions=positions)

        if self.flatten:
            query = query.reshape(x.shape[0], -1, x.shape[-1])
        return query


class FourierFeatures(nn.Module):
    """Sinusoidal positional encodings for scalar data.

    See ["Fourier Features Let Networks Learn High Frequency Functions in Low
    Dimensional Domains"](https://arxiv.org/abs/2006.10739).

    Args:
        coef: geometric frequency coefficient / scale.
        features: number of features to use, evenly split between sin and cos;
            must be even.
    """

    def __init__(
        self, features: int = 16, coef: float = 10000.
    ) -> None:
        super().__init__()
        self.coef = coef
        self.features = features

        if features % 2 != 0:
            raise ValueError(f"Number of features must be even: {features}.")

    def forward(
        self, x: Float[Tensor, "*shape"]
    ) -> Float[Tensor, "*shape features"]:
        """Apply Fourier features.

        Args:
            x: input data; each element should be a scalar value.

        Returns:
            Data with Fourier features applied, adding a trailing features
                axis.
        """
        nc = self.features // 2
        w = self.coef ** (-torch.arange(nc, device=x.device) / nc)

        broadcast = [None] * (len(x.shape) - 1) + [slice(None)]
        wt = (x[..., None] * w[tuple(broadcast)])

        return torch.concatenate(
            [torch.cos(2 * torch.pi * wt), torch.sin(2 * torch.pi * wt)],
            dim=-1)
