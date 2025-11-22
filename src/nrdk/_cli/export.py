"""Export model weights from a full-service checkpoint."""

import os
import re

import torch
import yaml
from omegaconf import OmegaConf


def cli_export(
    path: str, /, output: str = "weights.pth", config: str = "model.yaml"
) -> None:
    """Export model weights from a full-service checkpoint.

    !!! info "Usage"

        Take the best checkpoint in `results/experiment/version`, and export
        the model to `results/experiment/version/weights.pth` and config to
        `results/experiment/version/model.yaml`:
        ```sh
        nrdk export results/experiment/version
        ```

    The model is assumed to be created by
    [`NRDKLightningModule`][nrdk.framework.] (via pytorch lightning), so has a
    `state_dict` attribute that contains the model weights, where each key
    has a leading `.model` prefix.

    - If the `path` points to a file, export that checkpoint.
    - If the `path` is a directory, the directory should have a
        `checkpoints.yaml` with a `best` key which specifies the best
        checkpoint in a `checkpoints/` directory; the exported model and config
        are saved relative to the `path`.

    !!! warning

        If `path` is a file (i.e., a specific checkpoint), the model config
        will not be exported (since its path not defined).

    Args:
        path: path to the checkpoint.
        output: path to save the exported weights.
        config: path to save the exported model config.
    """
    if os.path.isdir(path):
        hydra_cfg = OmegaConf.load(os.path.join(path, ".hydra", "config.yaml"))
        resolved = OmegaConf.to_container(hydra_cfg, resolve=True)
        assert isinstance(resolved, dict)
        # Save to config file
        with open(os.path.join(path, config), 'w') as f:
            yaml.dump({
                "model": resolved["model"],
                "transforms": resolved["transforms"]
            }, f, default_flow_style=False, sort_keys=False)

        output = os.path.join(path, output)
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
