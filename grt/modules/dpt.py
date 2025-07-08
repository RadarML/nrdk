"""Blocks for DPT."""

from beartype.typing import Literal
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn

from .conv import ConvResidual, ConvSeparable


class Fusion2D(nn.Module):
    """2D 2x upsampling fusion block.

    Similar to [M7]_, performs 3x3 (residual) convolutions on the encoder
    input `x_enc` added to the decoder input `x`, which is upsampled. To make
    it play nice with the hierarchical structure of swin transformers, we add
    an optional linear projection to reduce the number of features at the end.

    Args:
        d_in, d_out: input and output channels. Note the encoder and decoder
            inputs must both have `d_in` channels. If `d_in == d_out`, no
            linear projection is performed.
        activation: activation function to use; must be a `nn.Module` (i.e. not
            `nn.functional.*`).
        conv_type: type of convolution to use; can be "full" (ordinary conv) or
            "separable" (depthwise separable convolutions).
    """

    def __init__(
        self, d_in: int, d_out: int, activation: str = "GELU",
        conv_type: Literal["full", "separable"] = "full",
    ) -> None:
        super().__init__()

        self.d_out = d_out
        conv_block = (ConvResidual if conv_type == "full" else ConvSeparable)
        self.conv1 = conv_block(d_in, activation=activation)
        self.conv2 = conv_block(d_in, activation=activation)

        self.project = nn.Conv2d(
            d_in, d_out * 2 * 2, kernel_size=(1, 1), bias=True)

    def forward(
        self, x: Float[Tensor, "n c h w"], x_enc: Float[Tensor, "n c h w"]
    ) -> Float[Tensor, "n c2 h2 w2"]:
        """Apply 2D patch fusion decoder.

        Args:
            x: decoder input.
            x_enc: encoder input; should already be in the same shape as `x`.

        Returns:
            Fused and upsampled output.
        """
        fused = self.conv2(x + self.conv1(x_enc))
        fused = self.project(fused)
        return rearrange(
            fused, "n (c dh dw) h w -> n c (h dh) (w dw)",
            c=self.d_out, dh=2, dw=2)


class FusionDecoder(nn.Module):
    """2D fusion block, with a change-of-basis twist.

    Unlike `Fusion2D`, `FusionDecoder` does not assume that `x` and `x_enc`
    have the same spatial dimensions; instead, an attention mechanism is
    applied with `x` as the query, and `x_enc` as the key/value.

    Args:
        d_in, d_out: input and output channels.
        n_head: number of decoder heads.
        dropout: dropout to use during training.
        activation: activation function to use; must be a `nn.Module` (i.e. not
            `nn.functional.*`).
        conv_type: type of convolution to use; can be "full" (ordinary conv) or
            "separable" (depthwise separable convolutions).
    """

    def __init__(
        self, d_in: int, d_out: int, n_head: int = 8, dropout: float = 0.0,
        activation: str = "GELU",
        conv_type: Literal["full", "separable"] = "full"
    ) -> None:
        super().__init__()

        self.attn = nn.MultiheadAttention(
            d_in, n_head, dropout=dropout, bias=True, batch_first=True)
        self.fusion = Fusion2D(
            d_in=d_in, d_out=d_out, activation=activation, conv_type=conv_type)

    def forward(
        self, x: Float[Tensor, "n c h w"],
        x_enc: Float[Tensor, "n t c"]
    ) -> Float[Tensor, "n c2 h2 w2"]:
        """Apply fusion decoder."""
        n, c, h, w = x.shape
        x_query = rearrange(x, "n c h w -> n (h w) c")
        x_cross = self.attn(x_query, x_enc, x_enc, need_weights=False)[0]
        x_cross = rearrange(x_cross, "n (h w) c -> n c h w", h=h, w=w)
        return self.fusion(x, x_cross)
