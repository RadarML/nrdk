"""Data loading and transforms for `roverd` datasets."""

from jaxtyping import install_import_hook

with install_import_hook("nrdk.roverd", "beartype.beartype"):
    from .dataloader import datamodule
    from .lidar import (
        Occupancy2D,
        Occupancy2DData,
        Occupancy3D,
        Occupancy3DData,
    )
    from .transforms import (
        Semseg,
        Spectrum,
        SpectrumData,
        Velocity,
        VelocityData,
    )

__all__ = [
    "datamodule",
    "Occupancy2D", "Occupancy2DData",
    "Occupancy3D", "Occupancy3DData",
    "Semseg",
    "Spectrum", "SpectrumData",
    "Velocity", "VelocityData",
]
