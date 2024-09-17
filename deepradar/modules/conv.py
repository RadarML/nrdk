"""Convolutional Utils."""

from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn


class ConvNeXTBlock(nn.Module):
    """Single ConvNeXT block [M1]_.

    Args:
        d_model: Model feature dimensions.
        d_bottleneck: Expansion feature dimension in the bottleneck layers;
            the expansion ratio is given by `d_bottleneck / d_model`.
        kernel_size: Conv kernel size; `kernel_size=7` is recommended by [M1]_.
        activation: Activation function to use.
    """

    def __init__(
        self, d_model: int = 64, d_bottleneck: int = 256, kernel_size: int = 7,
        activation: str = 'ReLU'
    ) -> None:
        super().__init__()

        self.dw = nn.Conv2d(
            in_channels=d_model, out_channels=d_model,
            kernel_size=(kernel_size, kernel_size), stride=1,
            padding='same', groups=d_model)
        self.norm = nn.LayerNorm(d_model, eps=1e-5, bias=True)
        self.pw1 = nn.Conv2d(
            d_model, d_bottleneck, kernel_size=(1, 1), stride=1)
        self.activation = getattr(nn, activation)()
        self.pw2 = nn.Conv2d(
            d_bottleneck, d_model, kernel_size=(1, 1), stride=1)

    def forward(self, x: Float[Tensor, "n c h w"]) -> Float[Tensor, "n c h w"]:
        """Apply ConvNeXT.

        Operates in NCHW (batch-feature-spatial) order.
        """
        x0 = x
        x = self.dw(x)
        x = rearrange(
            self.norm(rearrange(x, "n c h w -> n h w c")),
            "n h w c -> n c h w")
        x = self.pw1(x)
        x = self.activation(x)
        x = self.pw2(x)
        return x0 + x


class ConvDownsample(nn.Module):
    """Downsampling block [M1]_.

    Args:
        d_in: input features.
        d_out: output features; nominally `d_in * 2`.
    """

    def __init__(self, d_in: int = 64, d_out: int = 128) -> None:
        super().__init__()

        self.norm = nn.LayerNorm(d_in, eps=1e-5)
        self.conv = nn.Conv2d(d_in, d_out, kernel_size=2, stride=2)

    def forward(
        self, x: Float[Tensor, "n c1 h1 w1"]
    ) -> Float[Tensor, "n c2 h2 w2"]:
        """Apply 2x spatial downsampling layer."""
        x_norm = rearrange(
            self.norm(rearrange(x, "n c h w -> n h w c")),
            "n h w c -> n c h w")
        return self.conv(x_norm)
