"""GRT Reference implementation training script."""

import hydra
import torch


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
    trainer.fit(model=lightningmodule, datamodule=datamodule)


if __name__ == "__main__":
    train()
