# Project Reference & Template

The GRT reference implementation uses a [hydra](https://hydra.cc/docs/intro/) + [pytorch lightning](https://lightning.ai/docs/pytorch/stable/)-based stack on top of the NRDK; use this reference implementation to get started on a new project.

!!! tip

    The included reference implementation can be run out-of-the-box:

    1. Obtain a copy of the [I/Q-1M](https://radarml.github.io/red-rover/iq1m/), and save it (or link it) to `nrdk/grt/data/`.
    2. Create a virtual environment in `nrdk/grt` with `uv sync`.
    3. Run with `uv run train.py`; see the hydra config files in `nrdk/grt/config/` for options.

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

2. Set up the `nrdk` dependency.

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

## Configuration Groups

Hydra is organized around "configuration groups", which are collections of related configuration files. The reference training script is organized around several configuration groups which are intended to be used in routine experiments, as well as some others which should not generally need to be modified (`hydra`, `lightningmodule`).

### `meta` &mdash; Metadata

```sh
uv run train.py meta.name=example_experiment meta.version=test_version_0 resume=path/to/previous/checkpoint.ckpt
```

Experiment metadata is configured using the `meta` config group, including an optional `resume` field which can be used to resume training from a previous checkpoint (with identical model settings).

!!! warning

    Make sure to set the `meta.name` and `meta.version` fields!

### `size` &mdash; Model Dimensions

```sh
uv run train.py size.d_model=256 size.d_feedforward=1024 size.nhead=4 size.enc_layers=3 size.dec_layers=3
```

Certain model dimensions can be configured globally using a `size` config group, which is referenced by other model configurations:

```yaml
size:
  d_model: 512
  d_feedforward: 2048
  nhead: 8
  enc_layers: 4
  dec_layers: 4
```

### `base` &mdash; Base Model

```sh
uv run train.py +base=occ3d_to_semseg
```

Load a base model using the specified configuration; see [`NRDKLightningModule.load_weights`][nrdk.framework.NRDKLightningModule.load_weights] for details about how to configure this behavior.

=== "Base &rarr; 2D Occupancy"

    ```sh
    --8<-- "grt/config/base/occ3d_to_occ2d.yaml"
    ```

=== "Base &rarr; Semseg"

    ```sh
    --8<-- "grt/config/base/occ3d_to_semseg.yaml"
    ```

=== "Base &rarr; Odometry"

    ```sh
    --8<-- "grt/config/base/occ3d_to_vel.yaml"
    ```

### `datamodule` &mdash; Dataloader

To configure the sensors to load:
```sh
uv run train.py sensors@datamodule.dataset.sensors=[radar,lidar,semseg,pose]
```

To configure which traces to include in the dataset:
```sh
uv run train.py traces@datamodule.traces=[bike,indoor,outdoor]
```

See [`nrdk.roverd.datamodule`][nrdk.roverd.datamodule] for more details about the dataloader configuration.

### `model` &mdash; Model Architecture

```sh
uv run train.py model=default
uv run train.py model/decoder=semseg
```

The model is just any `torch.nn.Module`; the included [`TokenizerEncoderDecoder`][nrdk.framework.TokenizerEncoderDecoder] is a good starting point.

!!! warning

    The reference configs are built around [`TokenizerEncoderDecoder`][nrdk.framework.TokenizerEncoderDecoder], with the tokenizer, encoder, and decoder as nested sub-configs.
    
    Note that these sub-configs must be specified as `model/decoder=...`, not `model.decoder=...`!

---

### `objective` &mdash; Training Objective

```sh
uv run train.py objective=lidar3d
```

Training objectives are expected to implement the the [`Objective`][abstract_dataloader.ext.objective] interface.

### `transforms` &mdash; Data Processing

```sh
uv run train.py transforms@transforms.sample=[radar,lidar3d]
```

### `lightningmodule` &mdash; Training Loop

The default `lightningmodule` config should not need to be modified, and pulls in the `${objective}` and `{$model}` configs.
