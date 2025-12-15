"""Training objectives.

Each objective is specified according to the [
`abstract_dataloader.ext.objective`][abstract_dataloader.ext.objective]
specifications. These objectives are designed to be assembled using a
[`MultiObjective`][abstract_dataloader.ext.objective.], with each objective
described by a [`MultiObjectiveSpec`][abstract_dataloader.ext.objective.].
Using a hydra-based system, the specification should follow this format:
```yaml
objectives:
  _target_: abstract_dataloader.ext.objective.MultiObjective
  objective_name:
    ... # objective config
```

??? quote "Hydra Configs"

    === "2D Lidar Occupancy"

        ```yaml title="objective/lidar2d.yaml"
        --8<-- "grt/config/objective/lidar2d.yaml"
        ```

    === "3D Lidar Occupancy"

        ```yaml title="objective/lidar3d.yaml"
        --8<-- "grt/config/objective/lidar3d.yaml"
        ```

    === "Camera Semantic Segmentation"

        ```yaml title="objective/semseg.yaml"
        --8<-- "grt/config/objective/semseg.yaml"
        ```

    === "Ego-Velocity"

        ```yaml title="objective/vel.yaml"
        --8<-- "grt/config/objective/vel.yaml"
        ```

    Note that you can combine multiple configurations to use multiple
    objectives, either in defaults:
    ```yaml
    defaults:
        - objective@lightningmodule.objective: ["lidar3d", "semseg"]
        ...
    ```
    or in the command line:
    ```sh
    uv run grt/train.py objective=[lidar3d,semseg] ...
    ```

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

    !!! info

        Each `*Data` type is described using protocols such that any classes
        which have the same attributes as the protocol can be used as inputs.
        For example, `SemsegData` can be replaced by any object (e.g., a
        dataclass) with a `semseg` attribute with dtype `UInt8` and shape
        `batch t h w`.
        ```python
        class SemsegData(Protocol, Generic[TArray]):
            semseg: UInt8[TArray, "batch t h w"]
        ```

- `YPred`: model output type, i.e., tensor shape and dtype.
"""

from nrdk._typecheck import typechecker

with typechecker("nrdk.objectives"):
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
