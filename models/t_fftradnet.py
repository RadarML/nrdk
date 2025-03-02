"""Radar swin transformer."""

from functools import partial

from beartype.typing import Optional, Sequence
from einops import rearrange
from jaxtyping import Float, Complex
from torch import Tensor, nn
import torch

from deepradar import modules


def flatten(x: Float[Tensor, "n *s c"]) -> Float[Tensor, "n flat c"]:
    """Collapse spatial dimensions."""
    return x.reshape(x.shape[0], -1, x.shape[-1])


class TFFTRadNetEncoder(nn.Module):
    """Radar Doppler Transformer.

    Args:
        stages: number of blocks in each stage; each block consists of one
            shifted and one un-shifted windowed attention module.
        dec_layers: which layers to pass to the decoder. If `None` (default),
            only the output of the last layer is returned.
        size: initial data size.
        patch: initial patch size.
        window: swin transformer window size.
        shift: amount to shift each window by.
        mask: whether each axis should be masked at boundaries during shifted
            windowed attention.
        merge: decimation factor to apply to each axis during patch merging
            at the end of each stage.
        dim: initial feature dimensions; must be divisible by `head_dim`.
            Each subsequent block doubles the number of features.
        head_dim: number of dimensions per head.
        ff_ratio: expansion ratio for feedforward blocks.
        dropout: dropout ratio during training.
        activation: activation function; specify as a name (i.e. corresponding
            to a class in `torch.nn`).
    """

    def __init__(
        self, stages: list[int] = [2, 2, 6, 2],
        dec_layers: Optional[Sequence[int]] = None,
        size: Sequence[int] = (64, 256),
        patch: Sequence[int] = (2, 2),
        window: Sequence[int] = (4, 4),
        shift: Sequence[int] = (2, 2),
        mask: Sequence[bool] = (False, True),
        merge: Sequence[int] = (2, 2),
        dim: int = 96, head_dim: int = 32, ff_ratio: float = 4.0,
        dropout: float = 0.1, activation: str = 'GELU',
    ) -> None:
        super().__init__()

        self.patch = modules.PatchMerge(
            d_in=3 * 4 * 2, d_out=dim, scale=patch, norm=False)
        self.pos = modules.Sinusoid()
        size = [s // p for s, p in zip(size, patch)]

        self.dec_layers = dec_layers
        self.stages = nn.ModuleList()
        self.merge = nn.ModuleList()
        for i, blocks in enumerate(stages):
            if i != 0:
                self.merge.append(modules.PatchMerge(
                    d_in=dim, d_out=dim * 2, scale=merge))
                size = [s // m for s, m in zip(size, merge)]
                dim *= 2

            swin_layer = partial(
                modules.SwinTransformerLayer,
                d_model=dim, n_head=dim // head_dim,
                d_feedforward=int(dim * ff_ratio), dropout=dropout,
                activation=activation, size=size, window=window)

            layers: list[nn.Module] = []
            for _ in range(blocks):
                layers.append(swin_layer(shift=None))
                layers.append(swin_layer(shift=shift, mask=mask))
            self.stages.append(nn.Sequential(*layers))


    def forward(
        self, x: Complex[Tensor, "n d a e r"]
    ) -> list[Float[Tensor, "n ?s ?c"]]:
        """Apply radar transformer.

        Args:
            x: input batch, with batch-doppler-azimuth-elevation-range-iq axis
                order.

        Returns:
            Encoding output.
        """
        x = rearrange(x, "n d a e r -> n d r (a e)")
        x = torch.fft.fft2(x, dim=(1, 2))
        x = rearrange(
            torch.stack([x.real, x.imag], dim=-1),
            "n d r ae cplx -> n d r (ae cplx)")

        x = self.pos(self.patch(x))

        out = []
        for stage, merge in zip(self.stages, self.merge):
            x = stage(x)
            out.append(x)
            x = merge(x)

        out.append(self.stages[-1](x))
        return out


class TFFTRadNet2DDecoder(nn.Module):

    def __init__(self, dim: int = 96, key: str = "bev") -> None:
        super().__init__()

        self.key = key
        self.L4  = nn.Conv2d(dim * 8, 256, kernel_size=1, stride=1, padding=0)
        self.L3  = nn.Conv2d(dim * 4, 256, kernel_size=1, stride=1, padding=0)
        self.L2  = nn.Conv2d(dim * 2, 256, kernel_size=1, stride=1, padding=0)

        self.up = nn.Upsample(
            scale_factor=(4, 4), mode='bilinear', align_corners=True)

        self.conv_out = nn.Sequential(
            nn.Conv2d(28, 64, (3, 3), padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3), padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1))

    def forward(self, encoded: list[Float[Tensor, "n ?s ?c"]]):

        T2: Float[Tensor, "n 16 64 256"]  = self.L2(
            rearrange(encoded[1], "n d r c -> n c d r"))
        T3: Float[Tensor, "n 8 32 256"]  = self.L3(
            rearrange(encoded[2], "n d r c -> n c d r"))
        T4: Float[Tensor, "n 4 16 256"]  = self.L4(
            rearrange(encoded[3], "n d r c -> n c d r"))

        stacked: Float[Tensor, "n 256 28 64"] = torch.concatenate([
            T2, torch.repeat_interleave(T3, 2, dim=3),
            torch.repeat_interleave(T4, 4, dim=3)
        ], dim=2)
        upsampled = self.up(rearrange(stacked, "n a e r -> n e a r"))
        res = self.conv_out(upsampled)
        return {self.key: res}
