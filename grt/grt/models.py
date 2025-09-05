"""GRT building blocks.

??? tip "Use `TransformerEncoder` as the GRT encoder"

    The GRT encoder is just a vanilla transformer encoder! If you're using
    hydra, use the following:
    ```yaml
    encoder:
      _target_: torch.nn.TransformerEncoder
      encoder_layer:
        _target_: torch.nn.TransformerEncoderLayer
        d_model: 512
        nhead: 8
        dim_feedforward: 2048
        dropout: 0.1
        activation: gelu
        layer_norm_eps: 1e-5
        batch_first: true
        norm_first: true
        bias: true
      num_layers: 4
      enable_nested_tensor: false
    ```
    To match the settings used by GRT, after selecting the appropriate
    `d_model` and `num_layers`:

    - Set `dim_feedforward` to `4.0 * d_model`
    - Set `n_head` to `d_model // 64`
"""

from collections.abc import Sequence
from typing import Literal

from jaxtyping import Float
from torch import Tensor, nn

from nrdk import modules
from nrdk.roverd import SpectrumData


class SpectrumTokenizer(nn.Module):
    """GRT 4D Radar Spectrum tokenizer.

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
        patch: input (doppler, azimuth, elevation, range) patch size.
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
        self, spectrum: SpectrumData
    ) -> Float[Tensor, "n s c"]:
        """Apply radar transformer.

        Args:
            spectrum: input batch spectrum data.

        Returns:
            Tokenized output.
        """
        x = spectrum.spectrum

        if self.squeeze is not None:
            x = self.squeeze(x)

        embedded = self.patch(x)

        if self.positions == "nd":
            embedded = self.pos(embedded)
        flat = embedded.reshape(embedded.shape[0], -1, embedded.shape[-1])
        if self.positions == "flat":
            flat = self.pos(flat)

        return self.readout(flat)
