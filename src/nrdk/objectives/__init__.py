"""Training objectives.

Each objective is specified according to the [
`abstract_dataloader.ext.objective`][abstract_dataloader.ext.objective]
specifications; for details about how to use objectives, see the
[`Objective`][abstract_dataloader.ext.objective.] specification.

!!! metrics

    In addition to a loss, each objective returns a set of metrics when called.

!!! visualizations

    Each objective can render a set of visualizations; though these can
    be empty in some case (e.g., `Velocity`).

Objectives also have three type parameters:

- `TArray`: the backend used; each objective provided is pytorch-based, so
    this is always [`torch.Tensor`][torch.Tensor].
- `YTrue`: ground truth data type; each objective class is accompanied by a
    corresponding `ObjectiveData` class which describes the expected input via
    a protocol type.
- `YPred`: model output type, i.e., tensor shape and dtype.

!!! info

    Each `*Data` type is described using protocols such that any classes which
    have the same attributes as the protocol can be used as inputs. For
    example, `SemsegData` can be replaced by any object (e.g., a dataclass)
    with a `semseg` attribute with dtype `UInt8` and shape `batch t h w`.
    ```python
    class SemsegData(Protocol, Generic[TArray]):
        semseg: UInt8[TArray, "batch t h w"]
    ```
"""

from jaxtyping import install_import_hook

with install_import_hook("nrdk.objectives", "beartype.beartype"):
    from .occupancy import (
        Occupancy2D,
        Occupancy2DData,
        Occupancy3D,
        Occupancy3DData,
    )
    from .odometry import Velocity, VelocityData
    from .semseg import Semseg, SemsegData

__all__ = [
    "Semseg", "SemsegData",
    "Occupancy3D", "Occupancy3DData",
    "Occupancy2D", "Occupancy2DData",
    "Velocity", "VelocityData"
]
