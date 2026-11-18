"""GRTv2 model wrapper."""

from collections.abc import Mapping
from typing import Any

from torch import nn


class GRTv2(nn.Module):
    """GRTv2 Model Framework.

    Unlike the original GRT implementation, the same decoder is used for all
    output heads with different query tokens; each head nominally only handles
    an output projection.

    Args:
        encoder: GRTv2 tokenizer/encoder module.
        decoder: GRTv2 decoder module.
        heads: output heads for the model.
        key: input data key for the spectrum data.
        mask_key: optional data key for an input mask.
        queries: Mapping of query keys to the corresponding data keys in the
            input; each key must correspond to a key in `heads`.
    """

    def __init__(
        self, encoder: nn.Module, decoder: nn.Module,
        heads: Mapping[str, nn.Module],
        key: str = "spectrum", mask_key: str | None = None,
        queries: Mapping[str, str] = {}
    ) -> None:
        super().__init__()

        if set(queries.keys()) != set(heads.keys()):
            raise ValueError(
                f"Keys of `queries` must match keys of `heads`. Found query "
                f"keys {set(queries.keys())} and head keys "
                f"{set(heads.keys())}.")

        self.encoder = encoder
        self.decoder = decoder
        self.key = key
        self.mask_key = mask_key
        self.queries = queries
        self.heads = nn.ModuleDict(heads)

    def _check_input_keys(self, data: dict[str, Any]) -> None:
        """Check that the input data contains the required keys."""
        if self.key not in data:
            raise KeyError(
                f"`GRTv2` is configured to expect an input key `{self.key}`, "
                f"but only the following are available\n:{list(data.keys())}")
        if self.mask_key is not None and self.mask_key not in data:
            raise KeyError(
                f"`GRTv2` is configured to expect a mask key "
                f"`{self.mask_key}`, but only the following are available\n:"
                f"{list(data.keys())}")
        for k_decoder, k_data in self.queries.items():
            if k_data not in data:
                raise KeyError(
                    f"Decoder query with key '{k_decoder}' expects input data "
                    f"key '{k_data}', but only the following are available\n:"
                    f"{list(data.keys())}")

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        """Forward pass through the GRTv2 model.

        In addition to input data (specified by `key`) and an optional mask
        (specified by `mask_key`), the input data dictionary should contain
        a query tensor for each head.

        Args:
            data: input data dictionary.

        Returns:
            Output data dictionary for each head.
        """
        self._check_input_keys(data)

        latents = self.encoder(
            data[self.key],
            mask=data[self.mask_key] if self.mask_key is not None else None)

        outputs = {}
        for head_name, head_model in self.heads.items():
            decoded = self.decoder(latents, data[self.queries[head_name]])
            output = head_model(decoded)

            if isinstance(output, dict):
                outputs.update(output)
            else:
                outputs[head_name] = output

        return outputs
