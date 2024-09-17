"""Blocks for DPT."""

from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn


class Fusion2D(nn.Module):
    """2D 2x upsampling fusion block.

    Similar to [M7]_, performs 3x3 (residual) convolutions on the encoder
    input `x_enc` added to the decoder input `x`, which is upsampled. However
    we do not apply a convolution to `x_enc`.

    Args:
        d_in, d_out: input and output channels. Note the encoder and decoder
            inputs must both have `d_in` channels.
        activation: activation function to use; must be a `nn.Module` (i.e. not
            `nn.functional.*`).
    """

    def __init__(
        self, d_in: int, d_out: int, activation: str = "GELU"
    ) -> None:
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out

        self.conv = nn.Conv2d(d_in, d_out, kernel_size=(3, 3), stride=1)
        self.activation = getattr(nn, activation)()

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
        x = x + x_enc
        x = x + self.activation(self.conv(x))
        return nn.functional.interpolate(
            x, scale_factor=2, mode="bilinear", align_corners=True)


class FusionDecoder(nn.Module):
    """2D fusion block, with a change-of-basis twist.

    Unlike `Fusion2D`, `FusionDecoder` does not assume that `x` and `x_enc`
    have the same spatial dimensions; instead, it's 

    Args:
        d_in, d_out: input and output channels.
        n_head: number of decoder heads.
        dropout: dropout to use during training.
        activation: activation function to use; must be a `nn.Module` (i.e. not
            `nn.functional.*`).
    """

    def __init__(
        self, d_in: int, d_out: int, n_head: int = 8, dropout: float = 0.0,
        activation: str = "GELU"
    ) -> None:
        super().__init__()

        self.attn = nn.MultiheadAttention(
            d_in, n_head, dropout=dropout, bias=True, batch_first=True)
        self.fusion = Fusion2D(d_in=d_in, d_out=d_out, activation=activation)

    def forward(
        self, x: Float[Tensor, "n c h w"],
        x_enc: Float[Tensor, "n t c2"]
    ) -> Float[Tensor, "n c h w"]:
        """Apply fusion decoder."""
        n, h, w, c = x.shape
        x_query = rearrange(x, "n c h w -> n (h w) c")
        x_cross = self.attn(x_query, x_enc, x_enc, need_weights=False)[0]
        x_cross = rearrange(x_cross, "n (h w) c -> n c (h w)", h=h, w=w)
        return self.fusion(x, x_cross)
