"""Swin/axial transformers."""

from beartype.typing import Literal, Optional, Sequence
from einops import einsum
from jaxtyping import Float
from torch import Tensor, nn

from .transformer import transformer_mlp
from .window import RelativePositionBias, WindowPartition


class WindowAttention(nn.Module):
    """N-dimensional windowed (self) attention.

    Loosly based on `WindowAttention` in [M6]_, and likewise implements both
    shifted and non-shifted windowed attention.

    Args:
        d_model: model embedding dimension.
        n_head: number of heads. Note that `d_model` must be divisible by
            `n_head`.
        dropout: dropout to apply to the attention mechanism.
        size: input embedding size; note that spatial dimensions should not
            be flattened.
        window: window size; must have the same length as `size`.
        shift: shift size; if `None`, uses a shift of `0` in all axes.
            Otherwise, must have the same length as `size`.
    """

    def __init__(
        self, d_model: int = 512, n_head: int = 8, dropout: float = 0.0,
        size: Sequence[int] = [], window: Sequence[int] = [],
        shift: Optional[Sequence[int]] = None
    ) -> None:
        super().__init__()

        if shift is None:
            shift = [0 for _ in size]

        if len(size) != len(window) or len(size) != len(shift):
            raise ValueError(
                f"Embedding size {size}, window size {window}, and shift size "
                f"{shift} must have the same number of dimensions.")
        if d_model % n_head != 0:
            raise ValueError(
                f"d_model={d_model} must be divisible by n_head={n_head}.")

        self.d_model = d_model
        self.n_head = n_head
        self.scale = (d_model // n_head) ** -0.5
        self.shift = shift
        self.window = WindowPartition(size=size, window=window)
        self.bias = RelativePositionBias(window=window, n_head=n_head)
        self.qkv = nn.Linear(d_model, d_model * 3, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: Float[Tensor, "n *t c"]) -> Float[Tensor, "n *t c"]:
        """Compute windowed self-attention.

        Axis abbreviations: (see implementation to audit)

        - `n`, `c`: batch, number of channels
        - `nw`: number of windows
        - `ws`: window size
        - `nh`: number of heads
        """
        n, *_, c = x.shape
        partitioned = self.window.partition(x)

        n_nw, ws, _ = partitioned.shape
        qkv = self.qkv(
            partitioned
        ).reshape(n_nw, ws, 3, self.n_head, c // self.n_head)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        q = q * self.scale

        attn_raw: Float[Tensor, "n_nw nh ws ws"] = einsum(
            q, k, "n_nw ws1 nh c, n_nw ws2 nh c -> n_nw nh ws1 ws2")
        bias: Float[Tensor, "nh ws ws"] = self.bias()
        mask: Float[Tensor, "nw ws ws"] = (
            -100. * self.window.attention_mask(self.shift))

        attn: Float[Tensor, "n nw nh ws ws"] = self.softmax(
            attn_raw.reshape(n, n_nw // n, self.n_head, ws, ws)
            + bias[None, None, :, :, :]
            + mask[None, :, None, :, :]
        ).reshape(n_nw, self.n_head, ws, ws)

        attended = einsum(
            self.dropout(attn), v,
            "n_nw nh ws1 ws2, n_nw ws2 nh c -> n_nw ws1 nh c"
        ).reshape(n_nw, ws, c)

        return self.window.unpartition(attended)


class SwinTransformerLayer(nn.Module):
    """Single N-dimensional swin transformer (encoder) layer.

    NOTE: as with conventional transformers, we only implement "pre-norm"
    [M3]_, [M4]_.

    Args:
        d_model: model embedding dimension.
        n_head: number of heads. Note that `d_model` must be divisible by
            `n_head`.
        d_feedforward: feedforward block hidden units.
        dropout: dropout to use during training.
        activation: activation function to use; must be a `nn.Module` (i.e. not
            `nn.functional.*`).
        size: input embedding size; note that spatial dimensions should not
            be flattened.
        window: window size; must have the same length as `size`.
        shift: shift size; if `None`, uses a shift of `0` in all axes.
            Otherwise, must have the same length as `size`.
    """

    def __init__(
        self, d_model: int = 512, n_head: int = 8, d_feedforward: int = 2048,
        dropout: float = 0.0, activation: str = "GELU",
        size: Sequence[int] = [], window: Sequence[int] = [],
        shift: Optional[Sequence[int]] = None
    ) -> None:
        self.attn = WindowAttention(
            d_model=d_model, n_head=n_head, dropout=dropout, size=size,
            window=window, shift=shift)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model, eps=1e-5, bias=True)
        self.feedforward = transformer_mlp(
            d_model=d_model, d_feedforward=d_feedforward,
            activation=activation, dropout=dropout)

    def forward(self, x: Float[Tensor, "n *t c"]) -> Float[Tensor, "n *t c"]:
        """Apply transformer; uses batch-spatial-feature order."""
        x = x + self.dropout(self.attn(self.norm(x)))
        x = x + self.feedforward(x)
        return x


class AxialTransformerLayer(nn.Module):
    """Single axial transformer layer (encoder) layer.

    Similar to [M8]_, we apply multiple attention operations with the specified
    window shapes (nominally alternating axes). Note that these operations can
    be combined using `compose` (i.e. `attn_1 o ... o attn_k(x)`) or `add`
    (i.e. `attn_1(x) + attn_2(x) + ... + attn_k(x)`).
    """

    def __init__(
        self, d_model: int = 512, n_head: int = 8, d_feedforward: int = 2048,
        dropout: float = 0.0, activation: str = "GELU",
        size: Sequence[int] = [], windows: Sequence[Sequence[int]] = [],
        shift: Optional[Sequence[int]] = None,
        mode: Literal["add", "compose"] = "compose"
    ) -> None:
        self.attn = [
            WindowAttention(
                d_model=d_model, n_head=n_head, dropout=dropout, size=size,
                window=w, shift=shift
            ) for w in windows]
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model, eps=1e-5, bias=True)
        self.mode = mode
        self.feedforward = transformer_mlp(
            d_model=d_model, d_feedforward=d_feedforward,
            activation=activation, dropout=dropout)

    def forward(self, x: Float[Tensor, "n *t c"]) -> Float[Tensor, "n *t c"]:
        """Apply transformer; uses batch-spatial-feature order."""
        xa = self.norm(x)

        if self.mode == "compose":
            for attn in self.attn:
                xa = attn(xa)
        else:  # mode == "add"
            xa = sum(attn(xa) for attn in self.attn)

        x = x + self.dropout(xa)
        x = x + self.feedforward(x)
        return x
