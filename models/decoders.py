"""Model Decoders."""


from torch import Tensor, nn
from jaxtyping import Float, PyTree


class Decoder(nn.Module):
    """Base class for decoders.
    
    Decoders take a nested structure (i.e. `PyTree`) as an input, and return a
    dict of `Tensor`s, where each key corresponds to a named output, e.g.
    `depth`, `occupancy`, etc.
    """

    def forward(
        self, encoded: PyTree[Float[Tensor, "..."]]
    ) -> dict[str, Float[Tensor, "..."]]:
        """Run decoder."""
        raise NotImplementedError()


class TransformerDecoder(nn.Module):
    """2D """

