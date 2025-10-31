"""Generic model architectures."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, TypeVar

from torch import nn

TReg = TypeVar("TReg", bound=dict[str, Any])

@dataclass
class Output[Tstage, Toutput]:
    """Module with "shortcut" outputs which are directly passed to the output.

    Type Parameters:
        - Tstage: type of the primary module output.
        - Toutput: type of the regularization side channel; must be a dictionary
            with string keys.

    Attributes:
        stage: output for this stage to pass to the next stage.
        output: outputs to pass directly to the model output.
    """

    stage: Tstage
    output: Toutput


class TokenizerEncoderDecoder(nn.Module):
    """Generic architecture with a tokenizer, encoder, and decoder(s).

    The `tokenizer` and `encoder` can optionally return a [`Regularized`][^.]
    output, in which case the regularization outputs (`reg`) are returned
    directly in the model output, with only the primary output (`out`) passed
    to the next stage in the model.

    !!! info

        The user/caller is responsible for ensuring that the dataloader output
        and model components are compatible (i.e., have the correct type).

    Args:
        tokenizer: tokenizer module.
        encoder: encoder module.
        decoder: decoder modules; each key corresponds to the output key.
        key: key in the input data to use as the model input.
        squeeze: eliminate non-temporal, non-batch singleton axes in the
            output of each decoder.
    """

    def __init__(
        self, tokenizer: nn.Module, encoder: nn.Module,
        decoder: Mapping[str, nn.Module],
        key: str = "spectrum", squeeze: bool = True
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.decoder = nn.ModuleDict(decoder)
        self.key = key
        self.squeeze = squeeze

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        """Forward pass through the model.

        Args:
            data: input data dictionary.

        Returns:
            Output data dictionary with the model's output.
        """
        if self.key not in data:
            raise KeyError(
                f"`TokenizerEncoderDecoder` was created with an expected "
                f"input key `{self.key}`, but only the following are "
                f"available\n:{list(data.keys())}")

        outputs = {}

        x = data[self.key]
        tokens = self.tokenizer(x)
        if isinstance(tokens, Output):
            outputs.update(tokens.output)
            tokens = tokens.stage

        encoded = self.encoder(tokens)
        if isinstance(encoded, Output):
            outputs.update(encoded.output)
            encoded = encoded.stage

        decoded = {k: v(encoded) for k, v in self.decoder.items()}

        # 0 - batch; 1 - temporal
        if self.squeeze:
            decoded = {
                k: v.squeeze(dim=tuple(range(2, v.ndim)))
                for k, v in decoded.items()}

        return {**outputs, **decoded}
