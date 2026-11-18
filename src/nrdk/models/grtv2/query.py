"""GRTv2 decoder query type."""

from typing import Generic

from abstract_dataloader.ext.types import TArray, dataclass
from jaxtyping import Float


@dataclass
class Query(Generic[TArray]):
    """GRTv2 output query.

    !!! info

        Each field is treated as a wildcard token when `None`; this replaces
        the positional encoding for each field with a learnable "wildcard"
        token.

    Attributes:
        t: relative timestamps in seconds.
        rng: range value in range bins.
        az: azimuth value in radians.
        el: elevation value in radians.
    """

    t: Float[TArray, "batch n"] | None
    rng: Float[TArray, "batch n"] | None
    az: Float[TArray, "batch n"] | None
    el: Float[TArray, "batch n"] | None
