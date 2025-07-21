"""Export model weights from a full-service checkpoint."""

import os
import re

import torch
import yaml


def cli_export(
    path: str, /, output: str = "weights.pth"
) -> None:
    """Export model weights from a full-service checkpoint.

    The model is assumed to be created by
    [`NRDKLightningModule`][nrdk.framework.] (via pytorch lightning), so has a
    `state_dict` attribute that contains the model weights, where each key
    has a leading `.model` prefix.

    - If the `path` points to a file, export that checkpoint.
    - If the `path` is a directory, the directory should have a
        `checkpoints.yaml` with a `best` key which specifies the best
        checkpoint in a `checkpoints/` directory; the exported model is saved
        relative to the `path`.

    Args:
        path: path to the checkpoint.
        output: path to save the exported weights.
    """
    if os.path.isdir(path):
        output = os.path.join(path, output)

    if os.path.isdir(path):
        try:
            with open(os.path.join(path, "checkpoints.yaml"), 'r') as f:
                contents = yaml.safe_load(f)
            path = os.path.join(path, "checkpoints", contents['best'])
        except FileNotFoundError:
            raise FileNotFoundError(f"No 'checkpoints.yaml' found in {path}.")

    contents = torch.load(path, map_location='cpu')
    if 'state_dict' not in contents:
        raise ValueError(f"Checkpoint {path} does not contain 'state_dict'.")

    print(f"Exporting: {path}")
    print(f"--> {output}")
    pattern = re.compile(r"^model\.")
    state_dict = {
        pattern.sub("", k): v for k, v in contents['state_dict'].items()}
    torch.save(state_dict, output)
