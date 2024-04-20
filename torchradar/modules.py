"""RadarML modules."""

import torch
from torch import nn, Tensor
from einops import rearrange

from beartype.typing import Optional
from jaxtyping import Float


class Rotary2D(nn.Module):
    """2D rotary encoding (RoPE).
    
    https://arxiv.org/abs/2104.09864

    Parameters
    ----------
    features: number of features in this embedding. 
    """

    def __init__(self, features: int = 512) -> None:
        super().__init__()
        self.d = features // 2

    def forward(
        self, x: Float[Tensor, "b x1 x2 f"]
    ) -> Float[Tensor, "b x1 x2 f"]:

        i = torch.arange(x.shape[3] // 4, device=x.device)
        theta = 100000 ** (-i / x.shape[1])
        m1 = torch.arange(x.shape[1], device=x.device)
        m2 = torch.arange(x.shape[2], device=x.device)

        def _get_R(m) -> Float[torch.Tensor, "x d 2 2"]:
            mt = m[:, None] * theta[None, :]
            cos = torch.cos(mt)
            sin = torch.sin(mt)
            R_stack = torch.stack(
                [torch.stack([cos, -sin]), torch.stack([sin, cos])])
            return torch.moveaxis(R_stack, (0, 1, 2, 3), (2, 3, 0, 1))

        x1: Float[torch.Tensor, "b x1 x2 d 2 1"] = (
            x[..., :self.d].reshape(*x.shape[:-1], -1, 2)[..., None])
        R1: Float[torch.Tensor, "1 x1  1 d 2 2"] = (
            _get_R(m1)[None, :, None, :, :, :])
        x1_emb = torch.matmul(R1, x1).reshape(*x.shape[:-1], -1)

        x2: Float[torch.Tensor, "b x1 x2 d 2 1"] = (
            x[..., self.d:].reshape(*x.shape[:-1], -1, 2)[..., None])
        R2: Float[torch.Tensor, "1 1  x2 d 2 2"] = (
            _get_R(m2)[None, None, :, :, :, :])
        x2_emb = torch.matmul(R2, x2).reshape(*x.shape[:-1], -1)

        return torch.concatenate([x1_emb, x2_emb], dim=3)


class Sinusoid(nn.Module):
    """Centered N-dimensional sinusoidal positional embedding."""

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self, x: Float[Tensor, "b ... f"]
    ) -> Float[Tensor, "b ... f"]:
        
        embedding_dim = x.shape[-1] // 2 // (len(x.shape) - 2)
        i = torch.arange(embedding_dim, device=x.device)
        theta = 100000 ** (-i / embedding_dim)

        def _embed(i, d):
            a = torch.arange(d, device=x.device) - d // 2
            embedding = torch.concatenate([
                torch.sin(theta[None, :] * a[:, None]),
                torch.cos(theta[None, :] * a[:, None])], dim=1)
            
            shape = list(x.shape[1:-1]) + [1]
            shape[i] = 1

            reshape = [None] * (len(x.shape) - 1)
            reshape[-1] = slice(None)  # type: ignore
            reshape[i] = slice(None)   # type: ignore

            tmp = torch.tile(embedding[reshape], shape)
            return tmp

        embedding = torch.concatenate(
            [_embed(i, d) for i, d in enumerate(x.shape[1:-1])], dim=-1)
        return embedding[None, ...] + x


class Patch2D(nn.Module):
    """Apply a linear patch embedding along two axes.

    Parameters
    ----------
    in_channels: number of input channels.
    features: number of output features; should be `>= in_channels * size**2`.
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
    
    Parameters
    ----------
    in_channels: number of input channels.
    features: number of output features; should be `>= in_channels * size**2`.
    size: patch size.
    """
        
    def forward(
        self, x: Float[Tensor, "xn xc x1 x2 x3 x4"]
    ) -> Float[Tensor, "xn x1_out x2_out x3 x4 xc_out"]:
        """Patch on axes x1, x2, keeping x3, x4 with a patch size of (1, 1)."""
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

    Parameters
    ----------
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
        embedding = self.linear(x)
        return self.unpatch(rearrange(embedding, "b (x1x2) c -> b c (x1x2)"))


class TransformerLayer(nn.Module):
    """Single transformer (encoder) layer.
    
    NOTE: we use only implement "pre-norm".
    https://github.com/pytorch/pytorch/issues/55270
    https://arxiv.org/pdf/2002.04745.pdf
    """

    def __init__(
        self, d_model: int = 512, n_head: int = 8, d_feedforward: int = 2048,
        dropout: float = 0.0, activation: Optional[nn.Module] = None
    ) -> None:
        super().__init__()

        if activation is None:
            activation = nn.GELU()

        self.attn = nn.MultiheadAttention(
            d_model, n_head, dropout=dropout, bias=True, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model, eps=1e-5, bias=True)

        self.feedforward = nn.Sequential(
            nn.LayerNorm(d_model, eps=1e-5, bias=True),
            nn.Linear(d_model, d_feedforward, bias=True),
            activation,
            nn.Dropout(dropout),
            nn.Linear(d_feedforward, d_model, bias=True),
            nn.Dropout(dropout))

        self.activation = activation

    def attention(self, x: Float[Tensor, "n t c"]) -> Float[Tensor, "n t c"]:
        x = self.norm(x)
        return self.dropout(self.attn(x, x, x, need_weights=False)[0])

    def forward(self, x: Float[Tensor, "n t c"]) -> Float[Tensor, "n t c"]:
        x = x + self.attention(x)
        x = x + self.feedforward(x)
        return x


class TransformerQuery(nn.Module):
    """Single "change-of-basis" query."""

    def __init__(
        self, d_model: int = 512, n_head: int = 8, queries: int = 256
    ) -> None:
        super().__init__()

        self.attn = nn.MultiheadAttention(
            d_model, n_head, dropout=0.0, bias=True, batch_first=True)
        self.norm = nn.LayerNorm(d_model, eps=1e-5, bias=True)
        self.embedding = nn.Parameter(
            torch.normal(0, 0.01, size=(queries, d_model)))

    def forward(self, x: Float[Tensor, "n t c"]) -> Float[Tensor, "n t2 c"]:
        x = self.norm(x)
        embedding = torch.tile(self.embedding[None, ...], (x.shape[0], 1, 1))
        return self.attn(embedding, x, x, need_weights=False)[0]
