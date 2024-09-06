"""RadarML modules.

.. [M1] RoFormer: Enhanced Transformer with Rotary Position Embedding
    https://arxiv.org/abs/2104.09864
.. [M2] ConvNeXt: A ConvNet for the 2020s
    https://arxiv.org/abs/2201.03545
.. [M3] On Layer Normalization in the Transformer Architecture
    https://arxiv.org/pdf/2002.04745.pdf
.. [M4] Issue @ pytorch relating to post-norm:
    https://github.com/pytorch/pytorch/issues/55270
"""

import torch
from beartype.typing import Iterable, Union
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


class Sinusoid(nn.Module):
    """Centered N-dimensional sinusoidal positional embedding."""

    def _embed(self, i, d, theta, x):
        """Former closure in `.forward`.

        You may be tempted to refactor this into a closure. Indeed, it was
        originally written as a closure on `theta` and `x`.

        Unfortunately, it turns out that **all closures which could
        hypothetically close on pytorch Tensors immediately result in a memory
        leak**. This is because closures create an `implicit reference cycle
        <https://pytorch.org/blog/understanding-gpu-memory-2/?hss_channel=tw-776585502606721024>`__,
        which causes the tensors not to be properly garbage collected when
        they should, and instead have to wait for the python cycle-detector.
        """
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

    def forward(
        self, x: Float[Tensor, "b ... f"]
    ) -> Float[Tensor, "b ... f"]:
        """Apply sinusoidal embedding.

        Data should be in batch-spatial-feature order.
        """
        embedding_dim = x.shape[-1] // 2 // (len(x.shape) - 2)
        i = torch.arange(embedding_dim, device=x.device)
        theta = 100000 ** (-i / embedding_dim)

        embedding = torch.concatenate([
            self._embed(i, d, theta, x)
            for i, d in enumerate(x.shape[1:-1])
        ], dim=-1)
        return embedding[None, ...] + x


class Rotary2D(nn.Module):
    """2D rotary encoding [M2]_.

    Args:
        features: number of features in this embedding.
    """

    def __init__(self, features: int = 512) -> None:
        super().__init__()
        self.d = features // 2

    def _get_R(self, m, theta) -> Float[torch.Tensor, "x d 2 2"]:
        """Closure on theta."""
        mt = m[:, None] * theta[None, :]
        cos = torch.cos(mt)
        sin = torch.sin(mt)
        R_stack = torch.stack(
            [torch.stack([cos, -sin]), torch.stack([sin, cos])])
        return torch.moveaxis(R_stack, (0, 1, 2, 3), (2, 3, 0, 1))

    def forward(
        self, x: Float[Tensor, "b x1 x2 f"]
    ) -> Float[Tensor, "b x1 x2 f"]:
        """Apply 2D rotary encoding.

        Data should be in batch-spatial-feature order.
        """
        i = torch.arange(x.shape[3] // 4, device=x.device)
        theta = 100000 ** (-i / x.shape[1])
        m1 = torch.arange(x.shape[1], device=x.device)
        m2 = torch.arange(x.shape[2], device=x.device)

        x1: Float[torch.Tensor, "b x1 x2 d 2 1"] = (
            x[..., :self.d].reshape(*x.shape[:-1], -1, 2)[..., None])
        R1: Float[torch.Tensor, "1 x1  1 d 2 2"] = (
            self._get_R(m1, theta)[None, :, None, :, :, :])
        x1_emb = torch.matmul(R1, x1).reshape(*x.shape[:-1], -1)

        x2: Float[torch.Tensor, "b x1 x2 d 2 1"] = (
            x[..., self.d:].reshape(*x.shape[:-1], -1, 2)[..., None])
        R2: Float[torch.Tensor, "1 1  x2 d 2 2"] = (
            self._get_R(m2, theta)[None, None, :, :, :, :])
        x2_emb = torch.matmul(R2, x2).reshape(*x.shape[:-1], -1)

        return torch.concatenate([x1_emb, x2_emb], dim=3)


class Learnable1D(nn.Module):
    """Learnable 1-dimensional positional embedding.

    Args:
        d_model: Model feature dimension.
        size: Number of positions; must be fixed (i.e. operates on fixed
            dimension input only).
    """

    def __init__(self, d_model: int = 512, size: int = 1024) -> None:
        super().__init__()
        self.embeddings = nn.Parameter(
            data=torch.normal(0, 0.02, (size, d_model)))

    def forward(self, x: Float[Tensor, "n #t c"]) -> Float[Tensor, "n t c"]:
        """Apply positional embedding.

        Data should be in batch-spatial-feature order.
        """
        return x + self.embeddings


class LearnableND(nn.Module):
    """Learnable N-dimensional positional embedding.

    Args:
        d_model: Model feature dimensions.
        shape: Shape of input positions; must be fixed.
    """

    def __init__(
        self, d_model: int = 512,
        shape: Union[list[int], tuple[int, ...]] = (16, 16)
    ) -> None:
        super().__init__()
        self.shape = shape
        self.embeddings = nn.ParameterList([
            torch.normal(0, 0.02, (d, d_model)) for d in shape])

    def forward(self, x: Float[Tensor, "n ... c"]) -> Float[Tensor, "n ... c"]:
        """Apply embeddings.

        Data should be in batch-spatial-feature order.
        """
        for i, e in enumerate(self.embeddings):
            idxs = [None] * (len(self.shape) + 1) + [slice(None)]
            idxs[i + 1] = slice(None)
            x = x + e[idxs]
        return x


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

        self.feedforward = nn.Sequential(
            nn.LayerNorm(d_model, eps=1e-5, bias=True),
            nn.Linear(d_model, d_feedforward, bias=True),
            getattr(nn, activation)(),
            nn.Dropout(dropout),
            nn.Linear(d_feedforward, d_model, bias=True),
            nn.Dropout(dropout))

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

        self.feedforward = nn.Sequential(
            nn.LayerNorm(d_model, eps=1e-5, bias=True),
            nn.Linear(d_model, d_feedforward, bias=True),
            getattr(nn, activation)(),
            nn.Dropout(dropout),
            nn.Linear(d_feedforward, d_model, bias=True),
            nn.Dropout(dropout))

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
        """Apply transformer; uses batch-spatial-feature order."""
        x = x + self.self_attention(x)
        x = x + self.cross_attention(x, x_enc)
        x = x + self.feedforward(x)
        return x


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
