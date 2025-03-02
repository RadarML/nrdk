"""Radar swin transformer."""

from functools import partial

from beartype.typing import Literal, Optional, Sequence
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn

from deepradar import modules


def flatten(x: Float[Tensor, "n *s c"]) -> Float[Tensor, "n flat c"]:
    """Collapse spatial dimensions."""
    return x.reshape(x.shape[0], -1, x.shape[-1])


class SwinTransformerEncoder(nn.Module):
    """Radar Doppler Transformer.

    Implementation notes:

    - All axes and shapes are specified in doppler-azimuth-elevation-range
      order, which corresponds to the order in which the data is acquired
      (slow time, TX, RX, fast time).
    - Unlike a normal vision swin transformer, we include a `mask` argument
      which denotes whether each axis should be interpreted as a interval
      (e.g. range, angle) or a modulus space (e.g. full 360-degree angle
      `phi mod 2pi*N` or doppler velocity `v mod N*Dmax`).

    Args:
        stages: number of blocks in each stage; each block consists of one
            shifted and one un-shifted windowed attention module.
        fa_layers: number of final "full attention" layers.
        dec_layers: which layers to pass to the decoder. If `None` (default),
            only the output of the last layer is returned.
        size: initial data size.
        patch: initial patch size.
        window: swin transformer window size.
        shift: amount to shift each window by.
        mask: whether each axis should be masked at boundaries during shifted
            windowed attention.
        merge: decimation factor to apply to each axis during patch merging
            at the end of each stage.
        dim: initial feature dimensions; must be divisible by `head_dim`.
            Each subsequent block doubles the number of features.
        head_dim: number of dimensions per head.
        ff_ratio: expansion ratio for feedforward blocks.
        dropout: dropout ratio during training.
        activation: activation function; specify as a name (i.e. corresponding
            to a class in `torch.nn`).
    """

    def __init__(
        self, stages: list[int] = [2, 2, 6, 2], fa_layers: int = 2,
        dec_layers: Optional[Sequence[int]] = None,
        size: Sequence[int] = (64, 256),
        patch: Sequence[int] = (2, 2),
        window: Sequence[int] = (4, 4, 2, 4),
        shift: Sequence[int] = (2, 2, 0, 2),
        mask: Sequence[bool] = (False, True, True, True),
        merge: Sequence[int] = (2, 1, 1, 2),
        dim: int = 64, head_dim: int = 32, ff_ratio: float = 4.0,
        dropout: float = 0.1, activation: str = 'GELU',
    ) -> None:
        super().__init__()

        self.patch = modules.PatchMerge(
            d_in=2, d_out=dim, scale=patch, norm=False)
        self.pos = modules.Sinusoid()
        size = [s // p for s, p in zip(size, patch)]

        self.dec_layers = dec_layers
        self.stages = nn.ModuleList()
        self.merge = nn.ModuleList()
        for blocks in stages:

            swin_layer = partial(
                modules.SwinTransformerLayer,
                d_model=dim, n_head=dim // head_dim,
                d_feedforward=int(dim * ff_ratio), dropout=dropout,
                activation=activation, size=size, window=window)

            layers: list[nn.Module] = []
            for _ in range(blocks):
                layers.append(swin_layer(shift=None))
                layers.append(swin_layer(shift=shift, mask=mask))
            self.stages.append(nn.Sequential(*layers))

            self.merge.append(modules.PatchMerge(
                d_in=dim, d_out=dim * 2, scale=merge))
            size = [s // m for s, m in zip(size, merge)]
            dim *= 2

        self.readout = modules.Readout(d_model=dim)
        self.final_stage = nn.Sequential(*[
            modules.TransformerLayer(
                d_model=dim, n_head=dim // head_dim,
                d_feedforward=int(dim * ff_ratio), dropout=dropout,
                activation=activation)
            for _ in range(fa_layers)])

    def forward(
        self, x: Float[Tensor, "n d a e r c"]
    ) -> list[Float[Tensor, "n ?s ?c"]] | Float[Tensor, "n ?s ?c"]:
        """Apply radar transformer.

        Args:
            x: input batch, with batch-doppler-azimuth-elevation-range-iq axis
                order.

        Returns:
            Encoding output.
        """
        x = self.pos(self.patch(x))

        out = []
        for stage, merge in zip(self.stages, self.merge):
            x = stage(x)
            out.append(flatten(x))
            x = merge(x)

        x = self.readout(flatten(x))
        x = self.final_stage(x)
        out.append(x)

        if self.dec_layers is None:
            return x
        else:
            return out


class SwinDPT2DDecoder(nn.Module):
    """Swin Transformer + Dense Pixel Transformer-style decoder."""

    def __init__(
        self, key: str, dim: int = 128, skip_factor: int = 2,
        head_dim: int = 64, stages: int = 3,
        dropout: float = 0.0, activation: str = "GELU",
        shape: Sequence[int] = (1024, 256), patch: Sequence[int] = (16, 16),
        out_dim: int = 0, conv_type: Literal["full", "separable"] = "full"
    ) -> None:
        super().__init__()

        dim = dim * skip_factor

        self.key = key
        self.out_dim = out_dim
        self.query = modules.BasisChange(shape=(
            shape[0] // patch[0] // 2**stages,
            shape[1] // patch[1] // 2**stages), flatten=False)

        self.head = modules.ConvResidual(
            dim // 2, kernel_size=(3, 3), activation="GELU")
        self.unpatch = modules.Unpatch(
            output_size=(shape[0], shape[1], max(1, self.out_dim)),
            features=dim // 2, size=patch)

        self.layers = nn.ModuleList([
            modules.FusionDecoder(
                d_in=int(dim * 2**i),
                d_out=int(dim * 2**(i - 1)),
                n_head=int(dim * 2**i) // head_dim, dropout=dropout,
                activation=activation, conv_type=conv_type)
            for i in reversed(range(stages))])

    def forward(
        self, encoded: list[Float[Tensor, "n ?s ?c"]]
    ) -> dict[str, Float[Tensor, "n h w ..."]]:
        """Apply decoder."""
        out = self.query(encoded[-1][:, -1, :])
        out = rearrange(out, "n h w c -> n c h w")
        for enc, layer in zip(reversed(encoded), self.layers):
            out = layer(out, enc)

        out = self.head(out)
        out = self.unpatch(rearrange(out, "n c h w -> n (h w) c"))
        if self.out_dim == 0:
            out = out[:, 0, :, :]

        return {self.key: out}
