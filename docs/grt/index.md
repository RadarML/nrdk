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
    import os
    from time import perf_counter
    from typing import Any

    import hydra
    import torch
    import yaml
    from lightning.pytorch import callbacks

    @hydra.main(version_base=None, config_path="./config", config_name="default")
    def train(cfg):
        """Train a model using the GRT reference implementation."""
        torch.set_float32_matmul_precision('medium')

        def _inst(path, *args, **kwargs):
            return hydra.utils.instantiate(cfg[path], *args, **kwargs)

        transforms = _inst("transforms")
        datamodule = _inst("datamodule", transforms=transforms)
        lightningmodule = _inst(
            "lightningmodule", model=_inst("model"),
            objective=_inst("objectives"), transforms=transforms)
        trainer = _inst("trainer")

        start = perf_counter()
        trainer.fit(model=lightningmodule, datamodule=datamodule)
        duration = perf_counter() - start

        meta: dict[str, Any] = {"duration": duration}
        for callback in trainer.callbacks:
            if isinstance(callback, callbacks.ModelCheckpoint):
                meta["best_k"] = {
                    k: v.item() for k, v in callback.best_k_models.items()}
                meta["best"] = callback.best_model_path
                break

        meta_path = os.path.join(trainer.logger.log_dir, "checkpoints.yaml")
        with open(meta_path, 'w') as f:
            yaml.dump(meta, f, sort_keys=False)

    if __name__ == "__main__":
        train()
    ```

??? quote "Minimal Training Script"

    ```python title="grt/train_minimal.py"
    import hydra
    import torch

    @hydra.main(version_base=None, config_path="./config", config_name="default")
    def train(cfg):
        def _inst(path, *args, **kwargs):
            return hydra.utils.instantiate(cfg[path], *args, **kwargs)

        transforms = _inst("transforms")
        datamodule = _inst("datamodule", transforms=transforms)
        lightningmodule = _inst(
            "lightningmodule", model=_inst("model"),
            objective=_inst("objectives"), transforms=transforms)
        trainer = _inst("trainer")
        trainer.fit(model=lightningmodule, datamodule=datamodule)

    if __name__ == "__main__":
        train()
    ```

Current train command (WIP):
```sh
uv run grt/train_minimal.py +objectives@objectives=lidar2d decoder@model.decoder=lidar2d sensors=[radar,lidar2d] aug@transforms.sample.augmentations=full
```
