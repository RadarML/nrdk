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
