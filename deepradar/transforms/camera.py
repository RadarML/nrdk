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
        self, data: Shaped[np.ndarray, "t h w *c"], aug: dict[str, Any] = {},
        idx: int = 0
    ) -> Shaped[np.ndarray, "t h w *c"]:
        """Apply camera-related augmentations.

        Args:
            data: stack of input frames, in batch-height-width-channel order.
            aug: data augmentations to apply.

        Returns:
            Stack of frames with augmentations applied uniformly.
        """
        if aug.get("azimuth_flip"):
            # A copy is required here since torch doesn't allow creating
            # tensors from data with negative stride.
            data = np.flip(data, axis=2).copy()
        return data
