"""Nonstandard 2D Convolutional modules."""

from collections.abc import Callable, Sequence
from typing import Literal

import torch
from jaxtyping import Float
from torch import Tensor, nn


class ConvNextLayer(nn.Module):
    """Residual convolutional layer following the ConvNext architecture.

    See ["A ConvNet for the 2020s"](https://arxiv.org/abs/2201.03545); each
    residual block consists of:

    1. A 7x7 depthwise convolution,
    2. Layer norm,
    3. A 1x1 "inverted bottleneck" with an expansion ratio of 4,
    4. GELU activation,
    5. And finally a 1x1 pointwise convolution.

    Args:
        channels: number of input/output channels in the model.
        expansion_ratio: expansion ratio for the inverted bottleneck.
        padding_mode: padding mode for the depthwise convolution; see
            [`torch.nn.Conv2d`][torch.nn.Conv2d] for details.
        layer_scale_init_value: initial value for layer scaling; if <= 0,
            layer scaling is disabled.
    """

    def __init__(
        self, channels: int, expansion_ratio: int | float = 4.0,
        padding_mode: Literal[
            'zeros', 'reflect', 'replicate', 'circular'] = "zeros",
        layer_scale_init_value: float = 1e-6
    ) -> None:
        super().__init__()

        d = int(channels * expansion_ratio)

        self.dw = nn.Conv2d(
            channels, channels, kernel_size=7, padding=3, groups=channels,
            padding_mode=padding_mode)
        self.norm = nn.LayerNorm(channels)
        self.act = nn.GELU()
        self.pw1 = nn.Conv2d(channels, d, kernel_size=1)
        self.pw2 = nn.Conv2d(d, channels, kernel_size=1)
        self.gamma = nn.Parameter(
            layer_scale_init_value * torch.ones((channels)), requires_grad=True
        ) if layer_scale_init_value > 0 else None

    def forward(self, x: Float[Tensor, "b c h w"]) -> Float[Tensor, "b c h w"]:
        """Forward pass through the ConvNext layer."""
        x1 = self.dw(x)
        x1 = self.norm(
            x1.permute(0, 2, 3, 1).contiguous()
        ).permute(0, 3, 1, 2).contiguous()
        x1 = self.pw2(self.act(self.pw1(x1)))
        if self.gamma is not None:
            x1 = self.gamma[None, :, None, None] * x1
        return x + x1


class ConvResidual(nn.Module):
    """[Resnet](https://arxiv.org/pdf/1603.05027) stage.

    Some modifications are applied:
    - Layer norm instead of batch norm.
    - GELU nonlinearity instead of ReLU.
    - Optional layer scale as in [ConvNeXt](https://arxiv.org/pdf/2201.03545).
    - Ability to specify padding mode for convolutional layers.

    !!! info "Axis Order"

        Tensors use channels-first `(batch, channels, height, width)` order,
        matching [`torch.nn.Conv2d`][torch.nn.Conv2d].

    Args:
        dim: channel dimension.
        padding_mode: padding mode for convolutional layers.
        layer_scale_init_value: initial value for layer scale; if <= 0, no
            layer scale is applied.
    """

    def __init__(
        self, dim: int,
        padding_mode: Literal[
            "zeros", "reflect", "replicate", "circular"] = "circular",
        layer_scale_init_value: float = 1e-6,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.conv1 = nn.Conv2d(
            dim, dim, kernel_size=3, padding=1, padding_mode=padding_mode)
        self.norm2 = nn.LayerNorm(dim)
        self.conv2 = nn.Conv2d(
            dim, dim, kernel_size=3, padding=1, padding_mode=padding_mode)

        self.act = nn.GELU()
        self.gamma = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True
        ) if layer_scale_init_value > 0 else None

    def forward(
        self, x: Float[Tensor, "n c h w"]
    ) -> Float[Tensor, "n c h w"]:
        x1 = self.norm1(
            x.permute(0, 2, 3, 1).contiguous()
        ).permute(0, 3, 1, 2).contiguous()
        x1 = self.act(x1)
        x1 = self.conv1(x1)

        x2 = self.norm2(
            x1.permute(0, 2, 3, 1).contiguous()
        ).permute(0, 3, 1, 2).contiguous()
        x2 = self.act(x2)
        x2 = self.conv2(x2)

        if self.gamma is not None:
            x2 = self.gamma[None, :, None, None] * x2

        return x + x2


class ConvDown(nn.Module):
    """Convolutional down-resolution (e.g., encoder) stage.

    Applies downsampling first, followed by a series of residual layers.

    !!! info "Axis Order"

        Tensors use channels-first `(batch, channels, height, width)` order,
        matching [`torch.nn.Conv2d`][torch.nn.Conv2d].

    Args:
        d_in: input channel dimension.
        d_out: output channel dimension.
        downsample: downsample factor; can be an int or a tuple of
            `(height, width)` downsample factors.
        depth: number of layers.
        layer: residual layer module to use with all arguments except
            the channel dimension bound.
    """

    def __init__(
        self, d_in: int, d_out: int, downsample: int | Sequence[int] = 2,
        depth: int = 6, layer: Callable[[int], nn.Module] = ConvResidual
    ) -> None:
        super().__init__()

        if not isinstance(downsample, int):
            downsample = tuple(downsample)
            if len(downsample) != 2:
                raise ValueError(
                    "Downsample must be an int or a sequence of 2 ints "
                    f"(height, width); got {len(downsample)}.")

        self.patch = nn.Conv2d(
            d_in, d_out, kernel_size=downsample,
            stride=downsample, bias=False)
        self.stem = nn.Sequential(*[layer(d_out) for _ in range(depth)])

    def forward(
        self, x: Float[Tensor, "n c h w"]
    ) -> Float[Tensor, "n c2 h2 w2"]:
        return self.stem(self.patch(x))


class ConvUp(nn.Module):
    """Convolutional up-resolution (e.g., decoder) stage.

    Applies a series of residual layers first, followed by upsampling.

    !!! info "Axis Order"

        Tensors use channels-first `(batch, channels, height, width)` order,
        matching [`torch.nn.Conv2d`][torch.nn.Conv2d].

    Args:
        d_in: input channel dimension.
        d_out: output channel dimension.
        upsample: upsample factor; can be an int or a tuple of
            `(height, width)` upsample factors.
        depth: number of layers.
        layer: residual layer module to use with all arguments except
            the channel dimension bound.
    """

    def __init__(
        self, d_in: int, d_out: int, upsample: int | Sequence[int] = 2,
        depth: int = 6, layer: Callable[[int], nn.Module] = ConvResidual
    ) -> None:
        super().__init__()

        if not isinstance(upsample, int):
            upsample = tuple(upsample)
            if len(upsample) != 2:
                raise ValueError(
                    "Upsample must be an int or a sequence of 2 ints "
                    f"(height, width); got {len(upsample)}.")

        self.stem = nn.Sequential(*[layer(d_in) for _ in range(depth)])
        self.unpatch = nn.ConvTranspose2d(
            d_in, out_channels=d_out,
            kernel_size=upsample, stride=upsample, bias=False)

    def forward(
        self, x: Float[Tensor, "n c h w"]
    ) -> Float[Tensor, "n c2 h2 w2"]:
        return self.unpatch(self.stem(x))


class ConvEncoder(nn.Sequential):
    """Convolutional encoder with multiple stages.

    !!! info "Axis Order"

        Tensors use channels-first `(batch, channels, height, width)` order.
        The channel dimension grows by 2x at each stage.

    Args:
        stages: depth of each stage.
        downsample: downsample factor for each stage; can be an int or a tuple
            of `(height, width)` downsample factors.
        d_in: input channel dimension.
        d_out: output channel dimension; if None, the output channel dimension
            is the same as the latent dimension.
        width: initial channel width; the channel width for stage `i` is
            `width * 2**(i - 1)`.
        layer: residual layer module to use with all arguments except
            the channel dimension bound.
    """

    def __init__(
        self, stages: Sequence[int] = (2, 2, 2),
        downsample: Sequence[int | Sequence[int]] | None = None,
        d_in: int = 24, width: int = 96, d_out: int | None = None,
        layer: Callable[[int], nn.Module] = ConvResidual
    ) -> None:
        if len(stages) == 0:
            raise ValueError("At least one stage is required.")

        if downsample is None:
            downsample = [1] + [2] * (len(stages) - 1)
        elif len(downsample) != len(stages):
            raise ValueError(
                f"Got {len(downsample)} downsample factors for "
                f"{len(stages)} stages; these must have the same length.")

        _stages = []
        for i, (ds, depth) in enumerate(zip(downsample, stages)):
            if i == 0:
                d1, d2 = d_in, width
            else:
                d1, d2 = width * 2**(i - 1), width * 2**i

            _stages.append(ConvDown(
                d_in=d1, d_out=d2, downsample=ds, depth=depth, layer=layer))

        self.latent_dim: int = width * 2**(len(stages) - 1)
        if d_out is not None:
            _stages.append(nn.Conv2d(self.latent_dim, d_out, kernel_size=1))

        super().__init__(*_stages)


class ConvDecoder(nn.Sequential):
    """Convolutional decoder with multiple stages.

    !!! info "Axis Order"

        Tensors use channels-first `(batch, channels, height, width)` order.
        The channel dimension shrinks by 2x at each stage.

    !!! warning

        `stages` and `upsample` are applied in reverse order to match
        the configuration for `ConvEncoder`.

    Args:
        stages: depth of each stage.
        upsample: upsample factor for each stage; can be an int or a tuple
            of `(height, width)` upsample factors.
        d_in: input channel dimension.
        width: final channel width; the channel width for stage `i` is
            `width * 2**(len(stages) - i - 1)`.
        layer: residual layer module to use with all arguments except
            the channel dimension bound.
    """

    def __init__(
        self, stages: Sequence[int] = (2, 2, 2),
        upsample: Sequence[int | Sequence[int]] | None = None,
        d_in: int = 24, width: int = 96,
        layer: Callable[[int], nn.Module] = ConvResidual
    ) -> None:
        if len(stages) == 0:
            raise ValueError("At least one stage is required.")

        if upsample is None:
            upsample = [1] + [2] * (len(stages) - 1)
        elif len(upsample) != len(stages):
            raise ValueError(
                f"Got {len(upsample)} upsample factors for "
                f"{len(stages)} stages; these must have the same length.")

        _stages = []
        _spec = list(enumerate(zip(upsample, stages)))
        for i, (us, depth) in reversed(_spec):
            if i == 0:
                d1, d2 = d_in, width
            else:
                d1, d2 = width * 2**(i - 1), width * 2**i

            _stages.append(ConvUp(
                d_in=d2, d_out=d1, upsample=us, depth=depth, layer=layer))

        super().__init__(*_stages)
        self.latent_dim: int = width * 2**(len(stages) - 1)
