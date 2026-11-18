"""Model zoo of reusable, stable models.

Models are organized into sub-namespaces by architecture:

- [`nrdk.models.grt`][nrdk.models.grt]: GRT tokenizers and decoders.
- [`nrdk.models.grtv2`][nrdk.models.grtv2]: GRTv2 encoder, decoder, and heads.

!!! warning

    Research models should not be placed here! Only stable (and
    nominally published) models should be implemented here.

    To add a new research model, create a new repository which uses the `nrdk`
    as a dependency; see the [reference implementation](../grt/index.md).
"""

from . import grt, grtv2

# Deprecated aliases: old configs use e.g.
# `_target_: nrdk.models.TransformerTensorDecoder`.
from .grt import MLPVectorDecoder as MLPVectorDecoder
from .grt import TransformerTensorDecoder as TransformerTensorDecoder

__all__ = ["grt", "grtv2"]
