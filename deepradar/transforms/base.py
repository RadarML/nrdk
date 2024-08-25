"""Base and generic classes."""

from abc import ABC, abstractmethod
from beartype.typing import Any
import numpy as np
from jaxtyping import Float, Float16


class Transform(ABC):
    """Generic transformation.

    args:
        path: dataset path to load any required metadata.
    """

    def __init__(self, path: str, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def __call__(self, data: Any, aug: dict[str, Any] = {}) -> Any:
        """Apply transform."""
        raise NotImplementedError()


class ToFloat16(Transform):
    """Convert to 16-bit float."""

    def __call__(
        self, data: Float[np.ndarray, "..."], aug: dict[str, Any] = {}
    ) -> Float16[np.ndarray, "..."]:
        return data.astype(np.float16)
