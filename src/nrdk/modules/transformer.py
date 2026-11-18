"""Transformer layers."""

from typing import Literal, overload

import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    !!! info "Axis Order"

        Normalization is applied along the last axis (the feature/channel
        dimension); the leading axes are unrestricted.

    Args:
        d_model: the number of expected features in the input.
        eps: a value added to the denominator for numerical stability.
    """

    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(
        self, x: Float[Tensor, "*batch channels"]
    ) -> Float[Tensor, "*batch channels"]:
        """Apply RMS normalization along the last axis."""
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class AATransformerEncoder(nn.Module):
    """Alternating global/frame attention transformer.

    Uses the architecture described by VGGT:

    - Includes alternating global and frame-wise attention layers,
        each implemented by a [`VisionTransformerEncoderLayer`][^.].
    - As the choice of global or frame first is arbitrary; like VGGT, we
        default to global-first.

    Note that this architecture also cleanly degrades to a standard transformer
    (i.e., as in GRT) when `t=1`.

    !!! info "Axis Order"

        Tensors use `(batch, frames, tokens, channels)` order. Global attention
        layers flatten `frames` and `tokens` into a single sequence; frame-wise
        attention layers process each frame independently.

    !!! warning

        The number of layers must be even due to the alternating attention
        pattern.

    Args:
        d_model: token embedding dimension.
        nhead: number of attention heads.
        dim_feedforward: feedforward dimension in transformer layers.
        dropout: dropout rate in transformer layers.
        layer_scale_init_value: initial value for layer scale; if <= 0, no
            layer scale is applied.
        layer_norm_eps: layer norm epsilon in transformer layers.
        bias: whether to use bias in transformer layers.
        num_layers: total number of transformer layers.
        norm: normalization type (layer or rms).
        activation: feedforward activation (gelu or swiglu).
    """

    def __init__(
        self, d_model: int = 512, nhead: int = 8, dim_feedforward: int = 2048,
        dropout: float = 0.1, layer_scale_init_value: float = 1e-6,
        layer_norm_eps: float = 1e-5, bias: bool = True, num_layers: int = 6,
        norm: Literal["layer", "rms"] = "layer",
        activation: Literal["gelu", "swiglu"] = "gelu",
    ) -> None:
        super().__init__()

        if num_layers % 2 != 0:
            raise ValueError(
                "Number of layers must be even for alternating attention.")

        self.layers = nn.ModuleList([
            VisionTransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                dropout=dropout, layer_norm_eps=layer_norm_eps, bias=bias,
                layer_scale_init_value=layer_scale_init_value, norm=norm,
                activation=activation)
            for _ in range(num_layers)
        ])

    @overload
    def forward(
        self, x: Float[Tensor, "n t s c"],
        output_attentions: Literal[False] = ...
    ) -> Float[Tensor, "n t s c"]: ...

    @overload
    def forward(
        self, x: Float[Tensor, "n t s c"], output_attentions: Literal[True]
    ) -> tuple[Float[Tensor, "n t s c"], Float[Tensor, "n t s s"]]: ...

    def forward(
        self, x: Float[Tensor, "n t s c"], output_attentions: bool = False,
    ) -> Float[Tensor, "n t s c"] | tuple[
        Float[Tensor, "n t s c"], Float[Tensor, "n t s s"]
    ]:
        """Forward pass through the alternating attention transformer.

        Args:
            x: input tensor of shape `(batch, frames, tokens, channels)`.
            output_attentions: if True, also return the self-attention weights
                from the final layer. Because the number of layers is always
                even, the final layer is always a frame-wise attention layer;
                its weights are returned with shape
                `(batch, frames, tokens, tokens)`.

        Returns:
            The output tensor, or a `(output, attention_weights)` tuple when
                `output_attentions=True`.
        """
        b, t, s, c = x.shape
        last_idx = len(self.layers) - 1

        attn_weights = None
        for i, layer in enumerate(self.layers):
            is_last = output_attentions and i == last_idx
            if i % 2 == 0:
                # Global attention
                xga = rearrange(x, "b t s c -> b (t s) c", t=t, s=s)
                xga = layer(xga)
                x = rearrange(xga, "b (t s) c -> b t s c", t=t, s=s)
            else:
                # Frame-wise attention
                xfa = rearrange(x, "b t s c -> (b t) s c", b=b, t=t)
                if is_last:
                    xfa, attn_weights = layer(xfa, output_attentions=True)
                    attn_weights = rearrange(
                        attn_weights, "(b t) s1 s2 -> b t s1 s2", b=b, t=t)
                else:
                    xfa = layer(xfa)
                x = rearrange(xfa, "(b t) s c -> b t s c", b=b, t=t)

        if output_attentions:
            assert attn_weights is not None
            return x, attn_weights
        return x


class VisionTransformerEncoderLayer(nn.Module):
    """Transformer encoder layer for vision/perception transformers.

    Has been modified to include many modern features, such as those listed in
    [Vit-5](https://arxiv.org/abs/2602.08071):

    - Pre-norm instead of post-norm
    - Optional layer scale (`layer_scale_init_value > 0`)
    - Support for [`RMSNorm`][^.] (`norm = "rms"`)

    !!! info "Axis Order"

        Tensors use `(batch, tokens, channels)` order (`batch_first=True`),
        matching [`torch.nn.MultiheadAttention`][torch.nn.MultiheadAttention].

    Args:
        d_model: the number of expected features in the input.
        nhead: the number of heads in the multiheadattention models.
        dim_feedforward: the dimension of the feedforward network model.
        dropout: the dropout value.
        layer_norm_eps: the eps value in layer normalization components.
        bias: whether to use bias in the linear layers.
        layer_scale_init_value: initial value for layer scale; if <= 0, no
            layer scale is applied.
        norm: normalization type (layer or rms).
        activation: feedforward activation (gelu or swiglu).
    """

    def __init__(
        self, d_model: int, nhead: int = 8, dim_feedforward: int = 2048,
        dropout: float = 0.1, layer_norm_eps: float = 1e-5, bias: bool = True,
        layer_scale_init_value: float = 1e-6,
        norm: Literal["layer", "rms"] = "layer",
        activation: Literal["gelu", "swiglu"] = "gelu",
    ) -> None:
        super().__init__()

        if norm == "layer":
            norm_layer = nn.LayerNorm
        elif norm == "rms":
            norm_layer = RMSNorm
        else:
            raise ValueError(f"Invalid norm type: {norm}")

        self.norm1 = norm_layer(d_model, eps=layer_norm_eps)
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, bias=bias, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.gamma1 = nn.Parameter(
            layer_scale_init_value * torch.ones((d_model)), requires_grad=True
        ) if layer_scale_init_value > 0 else None

        if activation not in ("gelu", "swiglu"):
            raise ValueError(f"Invalid activation type: {activation}")
        self._activation = activation

        self.norm2 = norm_layer(d_model, eps=layer_norm_eps)
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias)
        if activation == "swiglu":
            self.linear1_gate = nn.Linear(d_model, dim_feedforward, bias=bias)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias)
        self.dropout2 = nn.Dropout(dropout)
        self.gamma2 = nn.Parameter(
            layer_scale_init_value * torch.ones((d_model)), requires_grad=True
        ) if layer_scale_init_value > 0 else None

    @overload
    def forward(
        self, x: Float[Tensor, "n s c"],
        output_attentions: Literal[False] = ...
    ) -> Float[Tensor, "n s c"]: ...

    @overload
    def forward(
        self, x: Float[Tensor, "n s c"], output_attentions: Literal[True]
    ) -> tuple[Float[Tensor, "n s c"], Float[Tensor, "n s s"]]: ...

    def forward(
        self, x: Float[Tensor, "n s c"], output_attentions: bool = False,
    ) -> Float[Tensor, "n s c"] | tuple[
        Float[Tensor, "n s c"], Float[Tensor, "n s s"]
    ]:
        """Forward pass through the transformer encoder layer.

        Args:
            x: input tensor of shape `(batch, tokens, channels)`.
            output_attentions: if True, also return the self-attention weights
                averaged over heads, as a tensor of shape
                `(batch, tokens, tokens)`.

        Returns:
            The output tensor, or a `(output, attention_weights)` tuple when
                `output_attentions=True`.
        """
        # Self attention block (norm first)
        x_norm = self.norm1(x)
        attn_out, attn_weights = self.self_attn(
            x_norm, x_norm, x_norm, need_weights=output_attentions)
        attn_out = self.dropout(attn_out)
        if self.gamma1 is not None:
            attn_out = attn_out * self.gamma1
        x = x + attn_out

        # Feed forward block (norm first)
        x_norm = self.norm2(x)
        if self._activation == "swiglu":
            ff_out = self.linear2(
                self.dropout1(torch.nn.functional.silu(
                    self.linear1_gate(x_norm)) * self.linear1(x_norm)))
        else:
            ff_out = self.linear2(
                self.dropout1(torch.nn.functional.gelu(self.linear1(x_norm))))
        ff_out = self.dropout2(ff_out)
        if self.gamma2 is not None:
            ff_out = ff_out * self.gamma2
        x = x + ff_out

        if output_attentions:
            return x, attn_weights
        return x
