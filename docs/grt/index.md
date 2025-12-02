# GRT Reference Implementation & Project Template

The GRT reference implementation uses a [hydra](https://hydra.cc/docs/intro/) + [pytorch lightning](https://lightning.ai/docs/pytorch/stable/)-based stack on top of the NRDK; use this reference implementation to get started on a new project.

!!! tip

    The included reference implementation can be run out-of-the-box:

    1. Obtain a copy of the [I/Q-1M](https://radarml.github.io/red-rover/iq1m/), and save it (or link it) to `nrdk/grt/data/`.
    2. Create a virtual environment in `nrdk/grt` with `uv sync`.
    3. Run with `uv run train.py`; see the hydra config files in `nrdk/grt/config/` for options.

!!! info "Pre-Trained Checkpoints"

    Pre-trained model checkpoints for the GRT reference implementation on the [I/Q-1M dataset](https://radarml.github.io/red-rover/iq1m/) can also be found [here](https://radarml.github.io/red-rover/iq1m/osn/#download-from-osn).

## Quick Start

1. Create a new repository, and copy the contents of the `grt/` directory:
```
example-project/
├── config/
│   ├── aug/
|   ...
├── grt/
│   ├── __init__.py
|   ...
├── pyproject.toml
├── train.py
└── train_minimal.py
```

    !!! tip

        Don't forget to change the `name`, `authors`, and `description`!

2. Set up the `nrdk` dependency (`nrdk[roverd] >= 0.1.5`).

    !!! warning "Required Extras"

        Make sure you include the `roverd` extra, which installs the following:

        - [`roverd`](https://radarml.github.io/red-rover/roverd/): a dataloader for data collected by the [red-rover system](https://radarml.github.io/red-rover/)
        - [`xwr`](https://radarml.github.io/xwr/): radar signal processing for TI mmWave radars

    If using `uv`, uncomment one of the corresponding lines in the supplied `pyproject.toml` (and comment out the included `nrdk = { path = "../" }` line):

    === "Via Github"

        ```toml
        [tool.uv.sources]
        nrdk = { git = "ssh://git@github.com/radarml/nrdk.git" }
        ```
    
    === "Via Submodule"
    
        After `git submodule add git@github.com:RadarML/nrdk.git`:
        ```toml
        [tool.uv.sources]
        nrdk = { path = "./nrdk" }
        ```

## Training Script

The GRT template includes reference training scripts which can be used for high level training and fine tuning control flow. You can use these scripts as-is, or modify them to suit your needs; where possible, stick to the same general structure to maintain compatibility.

??? quote "Reference Training Script"

    ```python title="grt/train.py"
    --8<-- "grt/train.py"
    ```

??? quote "Minimal Training Script"

    ```python title="grt/train_minimal.py"
    --8<-- "grt/train_minimal.py"
    ```

## Evaluation Script

::: evaluate.evaluate
    options:
        show_root_heading: false
