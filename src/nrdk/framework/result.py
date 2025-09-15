"""Helpers for managing results files."""

import os
from functools import cached_property
from typing import Literal, overload

import yaml
from omegaconf import DictConfig, OmegaConf


class Result:
    """Helper class for managing results.

    !!! info

        Assumes the following structure:
        ```
        experiment_result_directory/
        ├── .hydra/
        |   ├── config.yaml
        |   ...
        ├── checkpoints/
        |   ├── epoch={i}-step={j}.ckpt
        |   ...
        ├── eval/
        |   ├── sample/
        |   |   ├── # evaluation for sample data
        |   ├── path/to/trace/
        |   |   ├── # evaluation metrics for each trace
        |   |   ...
        |   ...
        ├── checkpoints.yaml
        └── events.out.tfevents...
        ```

    Args:
        path: path to results directory.
    """

    def __init__(self, path: str) -> None:
        self.path = path

    @overload
    def config(self, omegaconf: Literal[True] = True) -> DictConfig: ...

    @overload
    def config(self, omegaconf: Literal[False]) -> dict: ...

    def config(self, omegaconf: bool = True) -> dict | DictConfig:
        """Load the configuration file used for this experiment.

        Args:
            omegaconf: if True, return an OmegaConf object (e.g., to resolve
                interpolations) instead of a plain dict.

        Returns:
            Loaded configuration dictionary or `DictConfig`.
        """
        with open(os.path.join(self.path, ".hydra", "config.yaml")) as f:
            cfg =yaml.safe_load(f)
        if omegaconf:
            cfg = OmegaConf.create(cfg)
            assert isinstance(cfg, DictConfig)
        return cfg

    @cached_property
    def _checkpoints(self) -> dict:
        with open(os.path.join(self.path, "checkpoints.yaml")) as f:
            return yaml.safe_load(f)

    @property
    def best(self) -> str:
        """Path to the best checkpoint."""
        return os.path.join(
            self.path, "checkpoints", self._checkpoints["best"])

    @property
    def checkpoints(self) -> dict[str, float]:
        """All checkpoints and their corresponding validation metrics."""
        return {
            os.path.join(self.path, v): k
            for k, v in self._checkpoints["best_k"].items()
        }
