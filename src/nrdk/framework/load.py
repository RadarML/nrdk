"""Load pre-trained model for inference."""

import os
from collections.abc import Callable

import hydra
import torch
import yaml
from torch.types import Storage


def load_model(
    path: str, config: str | None = None, freeze: bool = True,
    eval: bool = True,
    map_location: Callable[[Storage, str], Storage] | torch.device | str
        | dict[str, str] | None = "cpu"
) -> torch.nn.Module:
    """Load a pre-trained model from a given path.

    If `path` is a directory, it should contain a `weights.pth` file and a
    `config.yaml` file, i.e., as saved by
    [`nrdk export`](../cli.md/#nrdk-export).

    Args:
        path: path to the model checkpoint or directory.
        config: if `path` is a file, provide the model configuration separately
            here.
        freeze: whether to freeze the model parameters.
        eval: whether the model should be set to eval mode.
        map_location: where to place the loaded weights; see
            [`torch.load`][torch.load].

    Returns:
        Loaded model.
    """
    if os.path.isdir(path):
        if config is None:
            config = os.path.join(path, "model.yaml")
        path = os.path.join(path, "weights.pth")

    if not os.path.isfile(path):
        raise FileNotFoundError(f"Model weights file not found: {path}")
    if config is None or not os.path.isfile(config):
        raise FileNotFoundError(f"Model config file not found: {config}")

    with open(config) as f:
        model_spec = yaml.safe_load(f)

    model = hydra.utils.instantiate(model_spec["model"])
    weights = torch.load(path, map_location=map_location)
    model.load_state_dict(weights, strict=True)

    if freeze:
        for param in model.parameters():
            param.requires_grad = False
    if eval:
        model.eval()

    return model
