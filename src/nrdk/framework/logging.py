"""Unified logging interface."""

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

import numpy as np
from jaxtyping import UInt8
from lightning.pytorch import loggers


@runtime_checkable
class LoggerWithImages(Protocol):
    """Protocol type for loggers which support image logging.

    !!! question "Why LoggerWithImages?"

        Since the different pytorch lightning loggers do not have a unified
        image logging interface, we must define a unified interface; loggers
        used with [`ADLLightningModule`][^^.] must implement this protocol
        in order to support image logging.
    """

    def log_images(
        self, images: Mapping[str, UInt8[np.ndarray, "h w c"]], step: int = 0
    ) -> None:
        """Log images to the logger.

        Args:
            images: mapping of image names to images; underlying loggers may
                have additional restrictions on key naming rules.
            step: current training step.
        """
        ...


class TensorflowLogger(loggers.TensorBoardLogger, LoggerWithImages):
    """Tensorflow logger."""

    def log_images(
        self, images: Mapping[str, UInt8[np.ndarray, "h w c"]], step: int = 0
    ) -> None:
        for name, image in images.items():
            self.experiment.add_image(name, image, step, dataformats="HWC")


class MLFlowLogger(loggers.MLFlowLogger, LoggerWithImages):
    """MLFlow logger.

    !!! bug

        Since MLFlow doesn't support nested images or images across steps,
        we flatten the image names and append the step:
        ```
        filename = f"{name.replace('/', '.')}.{step}"
        ```
    """

    def log_images(
        self, images: Mapping[str, UInt8[np.ndarray, "h w c"]], step: int = 0
    ) -> None:
        # MLFlow doesn't support nested or stepped images, so here we go...
        for name, image in images.items():
            filename = f"{name.replace('/', '.')}.{step}"
            self.experiment.log_image(self.run_id, image, key=filename)
