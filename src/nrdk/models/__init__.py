"""Model zoo of reusable, stable models.

!!! warning

    Research models should not be placed here! Only stable (and
    nominally published) models should be implemented here.

    To add a new research model, create a new repository which uses the `nrdk`
    as a dependency; see the [reference implementation](../grt/index.md).
"""

from .grt import MLPVectorDecoder, TransformerTensorDecoder

__all__ = ["TransformerTensorDecoder", "MLPVectorDecoder"]
