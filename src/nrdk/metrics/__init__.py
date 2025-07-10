"""Reusable metrics beyond what is included in standard libraries."""

from jaxtyping import install_import_hook

with install_import_hook("nrdk.metrics", "beartype.beartype"):
    from .classification import BCE, FocalLoss
    from .metrics import (
        DepthWithConfidence,
        Lp,
        VoxelDepth,
        lp_power,
        mean_with_mask,
    )
    from .pointcloud import PolarChamfer2D, PolarChamfer3D

__all__ = [
    "BCE", "FocalLoss",
    "DepthWithConfidence", "Lp", "VoxelDepth", "lp_power", "mean_with_mask",
    "PolarChamfer2D", "PolarChamfer3D",
]
