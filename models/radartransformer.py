"""First prototype of the "Radar Transformer"."""

import torch
from torch import Tensor
from einops import rearrange
from radar import modules
from torch import nn

from beartype.typing import Sequence, Union
from jaxtyping import Float

from radar import modules


class BasisChange(nn.Module):
    """Create "change-of-basis" query.
    
    Uses a 'reference vector', e.g. the output for a readout token or the
    token-wise mean of the output.

    Parameters
    ----------
    shape: query shape.
    """

    def __init__(
        self, shape: Union[list[int], tuple[int, ...]] = (16, 16)
    ) -> None:
        super().__init__()

        self.pos = modules.Sinusoid()
        self.shape = shape

    def forward(self, x: Float[Tensor, "n c"]) -> Float[Tensor, "n t2 c"]:

        idxs = [slice(None)] + [None] * len(self.shape) + [slice(None)]
        query = self.pos(
            torch.tile(x[idxs], (1, *self.shape, 1))
        ).reshape(x.shape[0], -1, x.shape[-1])

        return query


class RadarTransformer(nn.Module):

    def __init__(
        self, dim: int = 768, ff_ratio: float = 4.0, heads: int = 12,
        dropout: float = 0.1, activation: str = 'GELU',
        out_shape: Sequence[int] = (1024, 256)
    ) -> None:
        super().__init__()

        self.out_shape = out_shape

        self.patch = modules.Patch4D(channels=2, features=dim, size=(16, 16))
        self.pos = modules.Sinusoid()

        self.readout = nn.Parameter(data=torch.normal(0, 0.02, (dim,)))

        self.encode = nn.ModuleList([
            modules.TransformerLayer(
                d_feedforward=int(ff_ratio * dim), d_model=dim, n_head=heads,
                dropout=dropout, activation=activation) for _ in range(3)])
        self.decode = nn.ModuleList([
            modules.TransformerDecoder(
                d_feedforward=int(ff_ratio * dim), d_model=dim, n_head=heads,
                dropout=dropout, activation=activation) for _ in range(3)])

        self.query = BasisChange(
            shape=(out_shape[0] // 16, out_shape[1] // 16))

        self.unpatch = modules.Unpatch2D(
            output_size=(out_shape[0], out_shape[1], 1),
            features=dim, size=(16, 16))
        self.activation = nn.Sigmoid()

    def forward(self, x):
        patch = self.patch(rearrange(x, "n d a e r c -> n c d r a e"))
        embedded = self.pos(patch)

        x0 = rearrange(embedded, "n d r a e c -> n (d r a e) c")
        readout = torch.tile(self.readout[None, None, :], (x0.shape[0], 1, 1))
        x0 = torch.concatenate([x0, readout], axis=1)  # type: ignore

        x1 = self.encode[0](x0)
        x2 = self.encode[1](x1)
        x3 = self.encode[2](x2)

        q = self.query(x3[:, -1, :])

        y3 = self.decode[2](q, x3)
        y2 = self.decode[1](y3, x2)
        y1 = self.decode[0](y2, x1)

        unpatch = self.unpatch(y1)[:, 0]
        return self.activation(unpatch)
