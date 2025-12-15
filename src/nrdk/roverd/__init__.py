"""Data loading and transforms for `roverd` datasets.

!!! warning

    The `roverd` dependency must be installed to use this submodule, either
    using the `nrdk[roverd]` extra or by manually installing the
    [`roverd` package](https://radarml.github.io/red-rover/roverd/).

Each `{Modality}` and `{Modality}Data` pair describe a data transform and the
corresponding data type which it generates. These transforms can be
individually instantiated and applied, or assembled into a
[`ADLDataModule`][abstract_dataloader.ext.lightning.] using the provided
[`nrdk.roverd.datamodule`][nrdk.roverd.datamodule] constructor.

???+ quote "Hydra Config for Individual Sensors"

    === "Lidar"

        ```yaml title="sensors/lidar.yaml"
        --8<-- "grt/config/sensors/lidar.yaml"
        ```

    === "Radar"

        ```yaml title="sensors/radar.yaml"
        --8<-- "grt/config/sensors/radar.yaml"
        ```

    === "Semseg"

        ```yaml title="sensors/semseg.yaml"
        --8<-- "grt/config/sensors/semseg.yaml"
        ```

    === "Pose"

        ```yaml title="sensors/pose.yaml"
        --8<-- "grt/config/sensors/pose.yaml"
        ```

??? quote "Sample Hydra Config"

    When instantiated, this config produces an
    [`ADLDataModule`][abstract_dataloader.ext.lightning.], with the `dataset`,
    `traces`, `transforms`, and `ptrain/pval` forwarded to the specified
    `datamodule` constructor.
    ```yaml
    datamodule:
      _target_: nrdk.roverd.datamodule
      dataset:
        _target_: roverd.Dataset.from_config
        _partial_: true
        sync:
          _target_: abstract_dataloader.generic.Next
          margin: [1.0, 1.0]
        reference: "radar"
        sensors:
          lidar:
            _partial_: true
            _target_: roverd.sensors.OSLidarDepth
            correction: auto
          radar:
            _partial_: true
            _target_: roverd.sensors.XWRRadar
            correction: auto
          _camera:
            _partial_: true
            _target_: roverd.sensors.Semseg
            correction: auto
          pose:
            _partial_: true
            _target_: roverd.sensors.Pose
            reference: radar
      traces:
        train:
        - list_of_training_traces
        - ...
      batch_size: 32
      samples: 8
      num_workers: 12
      prefetch_factor: 2
      subsample:
        val: 16384
      transforms: null  # or provide a transform specification
      ptrain: 0.8
      pval: 0.2
    ```
"""

from nrdk._typecheck import typechecker

with typechecker("nrdk.roverd"):
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
