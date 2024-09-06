"""Camera transforms."""

import numpy as np
from beartype.typing import Any
from jaxtyping import Shaped

from .base import Transform


class CameraAugmentations(Transform):
    """Handle camera-related augmentations."""

    def __call__(
        self, data: Shaped[np.ndarray, "h w ..."], aug: dict[str, Any] = {}
    ) -> Shaped[np.ndarray, "h w ..."]:
        if aug.get("azimuth_flip"):
            data = np.flip(data, axis=1).copy()
        return data
