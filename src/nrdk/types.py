"""Protocol types for defining interoperable interfaces."""

from typing import Protocol, runtime_checkable

from jaxtyping import Float
from torch import Tensor


@runtime_checkable
class SpectrumData(Protocol):
    """Spectrum data with two angular and/or spatial axes.

    Attributes:
        spectrum: spectrum tensor; `x1, x2` can be any angular or antenna
            representation, i.e., `tx-rx` or `el-az`.
    """

    spectrum: Float[Tensor, "batch t doppler x1 x2 rng ch"]
