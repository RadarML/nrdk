"""Load pre-trained model for inference."""

import os

import hydra
import torch
import yaml


def load_model(
    path: str, config: str | None = None, freeze: bool = True
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
    weights = torch.load(path)
    model.load_state_dict(weights, strict=True)

    if freeze:
        for param in model.parameters():
            param.requires_grad = False

    return model
