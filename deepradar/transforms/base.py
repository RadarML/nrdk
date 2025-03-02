"""Base and generic classes."""

from abc import ABC, abstractmethod

import numpy as np
from beartype.typing import Any
from jaxtyping import Float, Float16, Shaped
from einops import rearrange


class Transform(ABC):
    """Generic transformation.

    args:
        path: dataset path to load any required metadata.
    """

    def __init__(self, path: str, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def __call__(
        self, data: Any, aug: dict[str, Any] = {}, idx: int = 0
    ) -> Any:
        """Apply transform.

        Args:
            data: input data.
            aug: data augmentation specifications; these augmentations are
                nominally applied uniformly to all input data.
            idx: source index of the data; can be used to fetch auxiliary
                information if required.

        Returns:
            Transformed data. No particular type, axis, or dimension
            requirements are imposed.
        """
        raise NotImplementedError()


class ToFloat16(Transform):
    """Convert to 16-bit float."""

    def __call__(
        self, data: Float[np.ndarray, "..."], aug: dict[str, Any] = {},
        idx: int = 0
    ) -> Float16[np.ndarray, "..."]:
        """Convert generic floating point to 16-bit floating point."""
        return data.astype(np.float16)


class Reshape(Transform):
    """Reshape array using an `einops` pattern."""

    def __init__(self, path: str, pattern: str) -> None:
        self.pattern = pattern

    def __call__(
        self, data: Shaped[np.ndarray, "..."], aug: dict[str, Any] = {},
        idx: int = 0
    ) -> Shaped[np.ndarray, "..."]:
        return rearrange(data, self.pattern)
