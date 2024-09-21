"""Radar swin transformer."""

from functools import partial

from beartype.typing import Sequence, TypedDict, Literal
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn

from deepradar import modules


class SwinStageSpec(TypedDict):
    """Swin stage specifications.

    Attributes:
        merge: patch merge size at the end of the stage.
        blocks: number of blocks, where one block is one unshifted window and
            one shifted window.
        window: window size.
        shift: shift size.
    """

    merge: tuple[int, int, int, int]
    blocks: int
    window: tuple[int, int, int, int]
    shift: tuple[int, int, int, int]


def flatten(x: Float[Tensor, "n *s c"]) -> Float[Tensor, "n flat c"]:
    """Collapse spatial dimensions."""
    return x.reshape(x.shape[0], -1, x.shape[-1])


class SwinTransformerEncoder(nn.Module):
    """Radar Doppler Transformer.

    Args:
        stages: stage specifications; see :py:class:`.SwinStageSpec`.
        fc_layers: number of final "full" attention layers.
        size: initial size, in doppler-azimuth-elevation-range order.
        dim: initial feature dimensions; must be divisible by `head_dim`.
            Each subsequent block doubles the number of features.
        head_dim: number of dimensions per head.
        ff_ratio: expansion ratio for feedforward blocks.
        dropout: dropout ratio during training.
        activation: activation function; specify as a name (i.e. corresponding
            to a class in `torch.nn`).
        patch: input (doppler, range) patch size.
    """

    def __init__(
        self, stages: list[SwinStageSpec] = [], fa_layers: int = 2,
        size: Sequence[int] = (64, 8, 2, 256),
        dim: int = 128, head_dim: int = 64, ff_ratio: float = 4.0,
        dropout: float = 0.1, activation: str = 'GELU',
        patch: Sequence[int] = (4, 1, 1, 8)
    ) -> None:
        super().__init__()

        self.patch = modules.PatchMerge(
            d_in=2, d_out=dim, scale=patch, norm=False)
        self.pos = modules.Sinusoid()
        size = [s // p for s, p in zip(size, patch)]

        self.stages = nn.ModuleList()
        self.merge = nn.ModuleList()
        for stage in stages:

            swin_layer = partial(
                modules.SwinTransformerLayer,
                d_model=dim, n_head=dim // head_dim,
                d_feedforward=int(dim * ff_ratio), dropout=dropout,
                activation=activation, size=size, window=stage["window"])

            layers: list[nn.Module] = []
            for _ in range(stage["blocks"]):
                layers.append(swin_layer(shift=None))
                layers.append(swin_layer(
                    shift=stage["shift"], mask=[False, True, True, True]))
            self.stages.append(nn.Sequential(*layers))

            self.merge.append(modules.PatchMerge(
                d_in=dim, d_out=dim * 2, scale=stage["merge"]))
            size = [s // m for s, m in zip(size, stage["merge"])]
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
    ) -> list[Float[Tensor, "n ?s ?c"]]:
        """Apply radar transformer.

        Args:
            x: input batch, with batch-doppler-azimuth-elevation-range-iq axis
                order.

        Returns:
            Encoding output; each tensor is the output of each stage.
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

        return list(reversed(out))


class SwinDPT2DDecoder(nn.Module):
    """Swin Transformer + Dense Pixel Transformer decoder."""

    def __init__(
        self, key: str, dim: int = 128, head_dim: int = 64, stages: int = 3,
        dropout: float = 0.0, activation: str = "GELU",
        shape: Sequence[int] = (1024, 256), patch: Sequence[int] = (16, 16),
        out_dim: int = 0, conv_type: Literal["full", "separable"] = "full"
    ) -> None:
        super().__init__()

        self.key = key
        self.out_dim = out_dim
        self.query = modules.BasisChange(shape=(
            shape[0] // patch[0] // (2**(stages - 1)),
            shape[1] // patch[1] // (2**(stages - 1))), flatten=False)
        self.unpatch = modules.Unpatch2D(
            output_size=(shape[0], shape[1], max(1, self.out_dim)),
            features=dim, size=patch)

        self.layers = nn.ModuleList([
            modules.FusionDecoder(
                d_in=int(dim * 2**i),
                d_out=int(dim * 2**(max(i - 1, 0))),
                n_head=int(dim * 2**i) // head_dim, dropout=dropout,
                activation=activation, conv_type=conv_type)
            for i in reversed(range(stages))])


    def forward(
        self, encoded: list[Float[Tensor, "n ?s ?c"]]
    ) -> dict[str, Float[Tensor, "n h w ..."]]:
        """Apply decoder."""
        out = self.query(encoded[0][:, -1, :])
        out = rearrange(out, "n h w c -> n c h w")
        for enc, layer in zip(encoded, self.layers):
            out = layer(out, enc)

        out = self.unpatch(rearrange(out, "n c h w -> n (h w) c"))
        if self.out_dim == 0:
            out = out[:, 0, :, :]

        return {self.key: out}
