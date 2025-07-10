"""Generic model architectures."""

from collections.abc import Mapping
from typing import Any

from torch import nn


class TokenizerEncoderDecoder(nn.Module):
    """Generic architecture with a tokenizer, encoder, and decoder(s).

    !!! info

        The user/caller is responsible for ensuring that the dataloader output
        and model components are compatible (i.e., have the correct type).

    Args:
        tokenizer: tokenizer module.
        encoder: encoder module.
        decoders: decoder modules; each key corresponds to the output key.
        key: key in the input data to use as the model input.
        squeeze: eliminate non-temporal, non-batch singleton axes in the
            output of each decoder.
    """

    def __init__(
        self, tokenizer: nn.Module, encoder: nn.Module,
        decoders: Mapping[str, nn.Module],
        key: str = "spectrum", squeeze: bool = True
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.decoders = nn.ModuleDict(decoders)
        self.key = key
        self.squeeze = squeeze

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        """Forward pass through the model.

        Args:
            data: input data dictionary.

        Returns:
            Output data dictionary with the model's output.
        """
        x = data[self.key]
        tokens = self.tokenizer(x)
        encoded = self.encoder(tokens)
        decoded = {k: v(encoded) for k, v in self.decoders.items()}

        # 0 - batch; 1 - temporal
        if self.squeeze:
            decoded = {
                k: v.squeeze(dim=tuple(range(2, v.ndim)))
                for k, v in decoded.items()}

        return decoded
