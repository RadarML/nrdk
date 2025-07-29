# GRT Reference Implementation

!!! warning "Required Extras"

    The GRT reference implementation requires the following extras:

    - [`roverd`](https://radarml.github.io/red-rover/roverd/): a dataloader for data collected by the [red-rover system](https://radarml.github.io/red-rover/)
    - [`xwr`](https://radarml.github.io/xwr/): radar signal processing for TI mmWave radars

    Install/run with with
    ```sh
    uv sync --extra xwr --extra roverd
    uv run --extra xwr --extra roverd train.py ...
    # or just
    uv sync --all-extras
    ```

??? quote "Reference Training Script"

    ```python title="grt/train.py"
    --8<-- "grt/train.py"
    ```

??? quote "Minimal Training Script"

    ```python title="grt/train_minimal.py"
    --8<-- "grt/train_minimal.py"
    ```

Current train command (WIP):
```sh
uv run grt/train_minimal.py +objectives@objectives=lidar2d decoder@model.decoder=lidar2d sensors=[radar,lidar2d] aug@transforms.sample.augmentations=full
```

```sh
uv run grt/train.py trainer=debug globals.d_feedforward=1024 model/decoder=semseg --cfg job
```