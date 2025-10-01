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

    !!! danger

        Since `.hydra` is technically a hidden folder, many file operations
        (e.g., `mv experiment_result/*`) will skip or hide this folder by
        default.

    Args:
        path: path to results directory.
        validate: check that the path exists, and that it matches the expected
            structure.
    """

    def __init__(self, path: str, validate: bool = True) -> None:
        self.path = path
        if validate:
            self.validate(path)

    @staticmethod
    def find(
        path: str, follow_symlinks: bool = False, strict: bool = True
    ) -> list[str]:
        """Find all results directories under the given path.

        Args:
            path: path to search under.
            follow_symlinks: if True, follow symlinks when searching.
            strict: if `True`, only return directories that pass validation;
                if `False`, return directories with any `.hydra` folder or
                `checkpoints.yaml` file.

        Returns:
            List of paths to results directories (that contain a
                `checkpoints.yaml` and `.hydra` folder.)
        """
        results = []
        for root, dirs, files in os.walk(path, followlinks=follow_symlinks):
            if strict:
                if ".hydra" in dirs and "checkpoints.yaml" in files:
                    results.append(root)
            else:
                if ".hydra" in dirs or "checkpoints.yaml" in files:
                    results.append(root)
        return results

    @staticmethod
    def validate(path: str) -> None:
        """Validate that the given path is a valid results directory.

        Raises:
            ValueError: if the path does not exist, or does not have the
                expected structure.
        """
        if not os.path.exists(path):
            raise ValueError(f"Result {path} does not exist.")
        if not os.path.exists(os.path.join(path, ".hydra", "config.yaml")):
            raise ValueError(
                f"Result {path} exists, but does not have an associated "
                f"hydra configuration {path}/.hydra/config.yaml.")
        if not os.path.exists(os.path.join(path, "checkpoints.yaml")):
            raise ValueError(
                f"Result {path} exists, does not have a checkpoint index "
                f"file {path}/checkpoints.yaml.")

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
