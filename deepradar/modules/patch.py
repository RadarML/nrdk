"""Patching and unpatching modules."""

import numpy as np
import torch
from beartype.typing import Iterable, Sequence
from einops import rearrange
from jaxtyping import Complex, Float
from torch import Tensor, fft, nn


class FFTLinear(nn.Module):
    """GPU version of `transform.FFTLinear`.

    Args:
        pad: azimuth padding; output has shape (tx * rx + pad).
        axes: axes to apply an FFT to; 0=doppler, 1=azimuth, 2=range.
    """

    def __init__(
        self, pad: int = 0, axes: Iterable[int] = (0, 1, 2)
    ) -> None:
        super().__init__()
        self.pad = pad
        self.axes = axes

    def forward(
        self, data: Complex[Tensor, "N D Tx Rx R"]
    ) -> Complex[Tensor, "N D A R"]:
        """Apply FFT on GPU (possibly with large padding).

        Data is provided in batch-slow-tx-rx-fast order, and returned in
        batch-doppler-azimuth-range order.
        """
        n, d, tx, rx, r = data.shape
        assert tx == 2, "Only 2-tx mode is supported."

        iq_dar = data.reshape(n, d, tx * rx, r)

        if self.pad > 0:
            zeros = torch.zeros(
                [n, d, self.pad, r], dtype=torch.complex64, device=data.device)
            iq_dar = torch.concatenate([iq_dar, zeros], dim=2)

        dar = fft.fftn(iq_dar, dim=[x + 1 for x in self.axes])
        dar_shf = fft.fftshift(
            dar, dim=[x + 1 for x in self.axes if x in (0, 1)])
        return dar_shf


class Patch2D(nn.Module):
    """Apply a linear patch embedding along two axes.

    Args:
        in_channels: number of input channels.
        features: number of output features; should be
            `>= in_channels * size**2`.
        size: patch size.
    """

    def __init__(
        self, channels: int = 1, features: int = 512,
        size: tuple[int, int] = (4, 4)
    ) -> None:
        super().__init__()
        self.size = size
        self.patch = nn.Unfold(kernel_size=size, stride=size)
        self.linear = nn.Linear(channels * size[0] * size[1], features)

    def forward(
        self, x: Float[Tensor, "xn xc x1 x2"]
    ) -> Float[Tensor, "xn x1_out x2_out xc_out"]:
        """Apply 2D patching.

        Takes batch-feature-spatial inputs, and returns batch-spatial-feature.
        """
        xn, xc, x1, x2 = x.shape
        x1_out = x1 // self.size[0]
        x2_out = x2 // self.size[1]

        patched = self.patch(x)
        embedded = self.linear(
            rearrange(patched, "n c x1x2 -> (n x1x2) c"))
        return rearrange(
            embedded, "(n x1 x2) c -> n x1 x2 c", n=xn, x1=x1_out, x2=x2_out)


class Patch4D(Patch2D):
    """Split data into patches along two axes, keeping the other two constant.

    This is necessary since Pytorch `nn.Unfold` only supports 2D data...

    Args:
        in_channels: number of input channels.
        features: number of output features; should be
            `>= in_channels * size**2`.
        size: patch size.
    """

    def forward(
        self, x: Float[Tensor, "xn xc x1 x2 x3 x4"]
    ) -> Float[Tensor, "xn x1_out x2_out x3 x4 xc_out"]:
        """Patch on axes x1, x2, keeping x3, x4 with a patch size of (1, 1).

        Takes batch-feature-spatial order, returns batch-spatial-feature.
        """
        xn, xc, x1, x2, x3, x4 = x.shape
        x1_out = x1 // self.size[0]
        x2_out = x2 // self.size[1]

        patched = self.patch(
            rearrange(x, "n c x1 x2 x3 x4 -> (n x3 x4) c x1 x2"))
        embedded = self.linear(
            rearrange(patched, "(nx3x4) c (x1x2) -> (nx3x4 x1x2) c"))
        return rearrange(
            embedded, "(n x3 x4 x1 x2) c -> n x1 x2 x3 x4 c",
            n=xn, x1=x1_out, x2=x2_out, x3=x3, x4=x4)


class Unpatch2D(nn.Module):
    """Unpatch data.

    Args:
        output_size: output 2D shape.
        features: number of input features; should be `>= size * size`.
        size: patch size as (width, height, channels).
    """

    def __init__(
        self, output_size: tuple[int, int, int] = (1024, 256, 1),
        features: int = 512, size: tuple[int, int] = (16, 16)
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(features, output_size[-1] * size[0] * size[1])
        self.unpatch = nn.Fold(output_size[:-1], kernel_size=size, stride=size)

    def forward(
        self, x: Float[Tensor, "n x1x2 c"]
    ) -> Float[Tensor, "n x1_out x2_out c_out"]:
        """Perform 2D unpatching.

        Operates in batch-spatial-feature order; spatial axes are flattened on
        the input, and unflattened in the output.
        """
        embedding = self.linear(x)
        return self.unpatch(rearrange(embedding, "b (x1x2) c -> b c (x1x2)"))


class PatchMerge(nn.Module):
    """Merge patches with normalization and nominally reduced projection.

    Args:
        d_in: embedding dimension.
        d_out: output dimension; nominally less than `d_model * prod(scale)`.
        scale: downsampling factor to apply in each axis.
    """

    def __init__(
        self, d_in: int, d_out: int, scale: Sequence[int] = []
    ) -> None:
        super().__init__()

        self.scale = scale
        d_merge = d_in * int(np.prod(scale))
        self.norm = nn.LayerNorm(d_merge)
        self.reduction = nn.Linear(d_merge, d_out, bias=False)

    def _merge(self, x: Float[Tensor, "n *t c"]) -> Float[Tensor, "n *t2 c2"]:
        """Perform patch merging."""
        n, *t, c = x.shape
        dims = sum(([d // s, s] for d, s in zip(t, self.scale)), start=[n])
        order = (
            [0] + [2 * i + 1 for i in range(len(self.scale))]
            + [2 * i + 2 for i in range(len(self.scale))] + [-1])
        t2 = [d // s for d, s in zip(t, self.scale)]
        return x.reshape(dims + [c]).permute(order).reshape(n, *t2, -1)

    def forward(self, x: Float[Tensor, "n *t c"]) -> Float[Tensor, "n *t2 c2"]:
        """Merge and project."""
        merged = self._merge(x)
        return self.reduction(self.norm(merged))
