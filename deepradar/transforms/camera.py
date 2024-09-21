"""Camera transforms."""

import numpy as np
from beartype.typing import Any
from jaxtyping import Shaped

from .base import Transform


class CameraAugmentations(Transform):
    """Handle camera-related augmentations.

    Augmentations:

        - `azimuth_flip`: flip the camera left/right.
    """

    def __call__(
        self, data: Shaped[np.ndarray, "h w ..."], aug: dict[str, Any] = {}
    ) -> Shaped[np.ndarray, "h w ..."]:
        if aug.get("azimuth_flip"):
            # A copy is required here since torch doesn't allow creating
            # tensors from data with negative stride.
            data = np.flip(data, axis=1).copy()
        return data
