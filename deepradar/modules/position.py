"""Positional Embeddings."""

import torch
from jaxtyping import Float
from torch import Tensor, nn


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
        shape: list[int] | tuple[int, ...] = (16, 16)
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
