"""Transformer blocks."""

from jaxtyping import Float
from torch import Tensor, nn


def transformer_mlp(
    d_model: int = 512, d_feedforward: int = 2048, activation: str = "GELU",
    dropout: float = 0.0, eps: float = 1e-5
) -> nn.Module:
    """Create transformer MLP.

    Consists of the following sequential layers::

        layernorm
        linear: d_model -> d_feedforward
        activation, dropout
        linear: d_feedforward -> d_model
        dropout

    Args:
        d_model: model dimension.
        d_feedforward: hidden dimension, i.e. `d_model * expansion_ratio`.
        activation: activation function (class in `torch.nn`).
        dropout: dropout rate.
        eps: layer norm epsilon.
    """
    return nn.Sequential(
        nn.LayerNorm(d_model, eps=eps, bias=True),
        nn.Linear(d_model, d_feedforward, bias=True),
        getattr(nn, activation)(),
        nn.Dropout(dropout),
        nn.Linear(d_feedforward, d_model, bias=True),
        nn.Dropout(dropout))


class TransformerLayer(nn.Module):
    """Single transformer (encoder) layer.

    NOTE: we use only implement "pre-norm" [M3]_, [M4]_.

    Args:
        d_model: model feature dimensions.
        n_head: number of heads.
        d_feedforward: feedforward block hidden units.
        dropout: dropout to use during training.
        activation: activation function to use; must be a `nn.Module` (i.e. not
            `nn.functional.*`).
    """

    def __init__(
        self, d_model: int = 512, n_head: int = 8, d_feedforward: int = 2048,
        dropout: float = 0.0, activation: str = "GELU"
    ) -> None:
        super().__init__()

        self.attn = nn.MultiheadAttention(
            d_model, n_head, dropout=dropout, bias=True, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model, eps=1e-5, bias=True)
        self.feedforward = transformer_mlp(
            d_model=d_model, d_feedforward=d_feedforward,
            activation=activation, dropout=dropout)

    def attention(self, x: Float[Tensor, "n t c"]) -> Float[Tensor, "n t c"]:
        x = self.norm(x)
        return self.dropout(self.attn(x, x, x, need_weights=False)[0])

    def forward(self, x: Float[Tensor, "n t c"]) -> Float[Tensor, "n t c"]:
        """Apply transformer; uses batch-spatial-feature order."""
        x = x + self.attention(x)
        x = x + self.feedforward(x)
        return x


class TransformerDecoder(nn.Module):
    """Single transformer (decoder) layer.

    Args:
        d_model: model feature dimensions.
        n_head: number of heads.
        d_feedforward: feedforward block hidden units.
        dropout: dropout to use during training.
        activation: activation function to use; must be a `nn.Module` (i.e. not
            `nn.functional.*`).
    """

    def __init__(
        self, d_model: int = 512, n_head: int = 8, d_feedforward: int = 2048,
        dropout: float = 0.0, activation: str = "GELU"
    ) -> None:
        super().__init__()

        self.attn = nn.MultiheadAttention(
            d_model, n_head, dropout=dropout, bias=True, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model, eps=1e-5, bias=True)

        self.attn2 = nn.MultiheadAttention(
            d_model, n_head, dropout=dropout, bias=True, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-5, bias=True)

        self.feedforward = transformer_mlp(
            d_model=d_model, d_feedforward=d_feedforward,
            activation=activation, dropout=dropout)

    def self_attention(
        self, x: Float[Tensor, "n t c"]
    ) -> Float[Tensor, "n t c"]:
        x = self.norm(x)
        return self.dropout(self.attn(x, x, x, need_weights=False)[0])

    def cross_attention(
        self, x: Float[Tensor, "n t c"], x_enc: Float[Tensor, "n t2 c"]
    ) -> Float[Tensor, "n t c"]:
        x = self.norm(x)
        return self.dropout(self.attn2(x, x_enc, x_enc, need_weights=False)[0])

    def forward(
        self, x: Float[Tensor, "n t c"], x_enc: Float[Tensor, "n t2 c"]
    ) -> Float[Tensor, "n t c"]:
        """Apply transformer; uses batch-spatial-feature order.

        Args:
            x: input embedding.
            x_enc: embedding from the encoder to attend to.

        Returns:
            Output embedding, with the same shape as `x`.
        """
        x = x + self.self_attention(x)
        x = x + self.cross_attention(x, x_enc)
        x = x + self.feedforward(x)
        return x
