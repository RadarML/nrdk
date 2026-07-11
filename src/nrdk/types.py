"""Protocol types for defining interoperable interfaces."""

from typing import Generic, Protocol, runtime_checkable

from abstract_dataloader.ext.types import TArray
from jaxtyping import Float


@runtime_checkable
class SpectrumData(Protocol, Generic[TArray]):
    """Spectrum data with two angular and/or spatial axes.

    Attributes:
        spectrum: spectrum tensor; `x1, x2` can be any angular or antenna
            representation, i.e., `tx-rx` or `el-az`.
        timestamps: timestamps for each spectrum frame.
    """

    spectrum: Float[TArray, "batch t doppler x1 x2 rng ch"]
    timestamps: Float[TArray, "batch t"]


@runtime_checkable
class HasTimestamps(Protocol, Generic[TArray]):
    """Protocol for data with timestamps."""

    timestamps: Float[TArray, "batch t"]
