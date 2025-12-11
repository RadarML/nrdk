"""Reusable metrics beyond what is included in standard libraries.

!!! info

    Each metric has a
    `__call__(y_true: YTrue, y_hat: YHat) -> Float[Tensor, "batch"]` interface;
    however, the input types and shapes may vary arbitrarily.
"""

from jaxtyping import install_import_hook

with install_import_hook("nrdk.metrics", "beartype.beartype"):
    from .classification import BCE, BinaryDiceLoss, FocalLoss
    from .metrics import (
        DepthWithConfidence,
        Lp,
        VoxelDepth,
        lp_power,
        mean_with_mask,
    )
    from .pointcloud import PolarChamfer2D, PolarChamfer3D

__all__ = [
    "BCE", "BinaryDiceLoss", "FocalLoss",
    "DepthWithConfidence", "Lp", "VoxelDepth", "lp_power", "mean_with_mask",
    "PolarChamfer2D", "PolarChamfer3D",
]
