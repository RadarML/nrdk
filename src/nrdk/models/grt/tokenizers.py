"""GRT tokenizers."""

from collections.abc import Sequence
from typing import Literal

from jaxtyping import Float
from torch import Tensor, nn

from nrdk import modules
from nrdk.types import SpectrumData


class SpectrumTokenizer(nn.Module):
    """GRT 4D Radar Spectrum tokenizer (basic linear patch embedding).

    Two types of positional embeddings are supported:

    - `nd` (recommended): n-dimensional embeddings, splitting the input
        features into `d` equal chunks encoding each axis separately.
    - `flat`: flattened positional embeddings, similar to the original ViT.

    !!! info

        We use a relative coordinate system for positional embeddings instead
        of absolute position indices, where each axis is scaled to `[-1, 1]` by
        default and scaled by `pos_scale` and `global_scale` factors; see
        [`modules.Sinusoid`][nrdk.] for details.

    !!! warning

        Any axes specified in `squeeze` must have a patch size which is equal
        to the input size along that axis.

    Args:
        d_model: model feature dimension.
        patch: input (time, doppler, elevation, azimuth, range) patch size.
        squeeze: eliminate these axes by moving them to the channel axis prior
            to patching; specified by index.
        n_channels: number of input channels; see [`xwr.nn`][xwr.nn].
        scale: position embedding scale.
        w_min: minimum frequency for sinusoidal position embeddings.
        positions: type of positional embedding.
    """

    def __init__(
        self, d_model: int = 768, patch: Sequence[int] = (1, 2, 2, 8, 4),
        squeeze: Sequence[int] = [], n_channels: int = 2,
        scale: Sequence[float] | float | None = None,
        w_min: Sequence[float] | float | None = 0.2,
        positions: Literal["flat", "nd"] = "nd",
    ) -> None:
        super().__init__()

        if len(patch) != 5:
            raise ValueError(
                f"Invalid patch size: {patch}; expected 5 dims "
                f"(time, doppler, elevation, azimuth, range)")

        if len(squeeze) > 0:
            self.squeeze = modules.Squeeze(dim=squeeze, size=patch)
            n_channels = n_channels * self.squeeze.n_channels
            patch = [p for i, p in enumerate(patch) if i not in squeeze]
        else:
            self.squeeze = None

        self.patch = modules.PatchMerge(
            d_in=n_channels, d_out=d_model, scale=patch, norm=False)

        self.positions = positions
        self.pos = modules.Sinusoid(scale=scale, w_min=w_min)
        self.readout = modules.Readout(d_model=d_model)

    def forward(
        self, spectrum: SpectrumData | Float[
            Tensor, "batch t doppler el az rng ch"]
    ) -> Float[Tensor, "n s c"]:
        """Apply tokenizer.

        Args:
            spectrum: input batch spectrum data, or the raw spectrum tensor.

        Returns:
            Tokenized output, with a readout token appended at the end.
        """
        if isinstance(spectrum, SpectrumData):
            x = spectrum.spectrum
        else:
            x = spectrum

        if self.squeeze is not None:
            x = self.squeeze(x)

        embedded = self.patch(x)

        if self.positions == "nd":
            embedded = self.pos(embedded)
        flat = embedded.reshape(embedded.shape[0], -1, embedded.shape[-1])
        if self.positions == "flat":
            flat = self.pos(flat)

        return self.readout(flat)


class SpectrumAngularMLPTokenizer(nn.Module):
    """4D Radar Spectrum tokenizer with an MLP angular encoding.

    Patches are always cut along the `(time, doppler, range)` axes; angular
    (or tx/rx) axes are squeezed into the channel axis, then encoded via a
    small MLP prior to a linear patch embedding.

    Args:
        d_model: model feature dimension.
        mlp: hidden dimensions of the per-bin MLP applied to the squeezed
            angular features; each layer is followed by a GELU activation.
        patch: input (time, doppler, range) patch size.
        angle_bins: input size along the two angular axes, i.e.
            (elevation, azimuth); these axes are squeezed entirely.
        n_channels: number of input channels; see [`xwr.nn`][xwr.nn].
        scale: position embedding scale.
        w_min: minimum frequency for sinusoidal position embeddings.
        positions: type of positional embedding.
    """

    def __init__(
        self, d_model: int = 768,
        mlp: Sequence[int] = [512, 512, 64],
        patch: Sequence[int] = (1, 2, 4),
        angle_bins: Sequence[int] = (2, 8), n_channels: int = 2,
        scale: Sequence[float] | float | None = None,
        w_min: Sequence[float] | float | None = 0.2,
        positions: Literal["flat", "nd"] = "nd",
    ) -> None:
        super().__init__()

        if len(patch) != 3:
            raise ValueError(
                f"Invalid patch size: {patch}; expected 3 dims "
                f"(time, doppler, range)")

        # Get rid of angular dims and put them into channels
        self.squeeze = modules.Squeeze(
            dim=(2, 3), size=[*patch[:2], *angle_bins, patch[2]])
        n_channels = n_channels * self.squeeze.n_channels

        _layers = []
        for d in mlp:
            _layers.append(nn.Linear(n_channels, d))
            _layers.append(nn.GELU())
            n_channels = d
        self.mlp = nn.Sequential(*_layers)

        # Final patch merge
        self.patch = modules.PatchMerge(
            d_in=n_channels, d_out=d_model, scale=patch, norm=False)

        self.positions = positions
        self.pos = modules.Sinusoid(scale=scale, w_min=w_min)
        self.readout = modules.Readout(d_model=d_model)

    def forward(
        self, spectrum: SpectrumData | Float[
            Tensor, "batch t doppler el az rng ch"]
    ) -> Float[Tensor, "n s c"]:
        """Apply tokenizer.

        Args:
            spectrum: input batch spectrum data, or the raw spectrum tensor.

        Returns:
            Tokenized output, with a readout token appended at the end.
        """
        if isinstance(spectrum, SpectrumData):
            x = spectrum.spectrum
        else:
            x = spectrum

        x = self.squeeze(x)
        # MLP accepts multiple batch dims
        x = self.mlp(x)

        embedded = self.patch(x)

        if self.positions == "nd":
            embedded = self.pos(embedded)
        flat = embedded.reshape(embedded.shape[0], -1, embedded.shape[-1])
        if self.positions == "flat":
            flat = self.pos(flat)

        return self.readout(flat)
