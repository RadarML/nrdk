"""Configuration-related utilities to simplify hydra configurations."""

import logging
import os
from collections.abc import Mapping, Sequence
from typing import Any

import hydra
import yaml
from hydra.experimental.callback import Callback
from omegaconf import DictConfig

Nested = Sequence[str] | Mapping[str, "Nested"]


def expand(path: str | None = None, **nested: Nested) -> list[str]:
    """Expand a nested sequence of mappings and lists into a flat list.

    Each level in the nested structure should be a mapping, except the last
    level, which should be a sequence of strings. Each mapping corresponds to
    a directory, while the inner list contains file name or path leaves.

    Args:
        path: base path to prepend to the file path.
        nested: a nested file-system-like structure.

    Returns:
        A flat list of file paths described by `nested`.
    """
    def _expand(nested: Nested, base: str | None = None):

        def _join(p: str) -> str:
            if base is not None:
                return os.path.join(base, p)
            else:
                return p

        if isinstance(nested, Sequence):
            return [_join(item) for item in nested]
        elif isinstance(nested, Mapping):
            return sum((_expand(v, _join(k)) for k, v in nested.items()), [])
        else:
            raise ValueError("Nested structure must be a sequence or mapping.")

    return _expand(nested, base=path)


class _SingletonRegistry:
    """Register an object as a singleton, by name."""

    def __init__(self) -> None:
        self.singletons: dict[str, Any] = {}

    def register(self, **kwargs: Any) -> dict[str, Any]:
        self.singletons.update(kwargs)
        return kwargs

    def __call__(self, name: str) -> Any:
        try:
            return self.singletons[name]
        except KeyError:
            raise KeyError(
                f"No singleton registered with name {name}. The following "
                f"objects are registered: \n{self.singletons}")


Singleton = _SingletonRegistry()
"""Singleton registry for shared objects.

Use this registry in the case that the same object needs to be injected as a
dependency to mutiple different constructors.

- When instantiating the object, pass it as a key-value pair to
  `Singleton.register`; multiple objects can be registered at the same time.
- When using the object, call `Singleton(...)` with the same key to retrieve
  the object.

!!! example

    To register an `ExampleClass` when using Hydra:

    ```yaml
    singletons:
      _target_: nrdk.framework.Singleton.register
      example:
        _target_: path.to.module.ExampleClass
    ```

    Then, to use the registered `backend` later:
    ```yaml
    ...
      example:
        _target_: nrdk.framework.Singleton
        name: example
    ```
"""


def inst_from(
    path: str, key: str | Sequence[str] | None
) -> Any:
    """Load and instantiate an object from an arbitrary configuration file.

    Args:
        path: path to the configuration yaml file.
        key: key or sequence of keys to locate the object specification in the
            configuration file. If `None`, the entire configuration is used.

    Returns:
        Instantiated object (via `hydra.utils.instantiate`).
    """
    with open(path) as f:
        spec = yaml.safe_load(f)

    if key is not None:
        if isinstance(key, str):
            spec = spec[key]
        else:
            for k in key:
                spec = spec[k]
    return hydra.utils.instantiate(spec)


class PreventHydraOverwrite(Callback):
    """Prevent Hydra from overwriting existing output directories.

    To use this callback, add it to the `hydra` configuration:
    ```yaml
    hydra:
      callbacks:
        prevent_overwrite:
          _target_: nrdk.config.PreventHydraOverwrite
    ```

    !!! warning

        In multi-GPU setups, the check is only performed on the main process
        (i.e., [`RANK=0`](https://docs.pytorch.org/docs/stable/elastic/run.html#environment-variables))
        since other workers are expected to see the same output directory.
    """

    def on_run_start(self, config: DictConfig, **kwargs: Any) -> None:
        # Rank environment variables are not reliably named
        # Read everything I can think of to be sure
        not_rank0 = (
            os.environ.get('RANK', '0') != '0' and
            os.environ.get('LOCAL_RANK', '0') != '0' and
            os.environ.get('GLOBAL_RANK', '0') != '0'
        )
        if not not_rank0:
            return

        output_dir = config.hydra.run.dir
        if os.path.exists(output_dir):
            logging.error(
                f"Aborting: output directory '{output_dir}' already exists!")
            exit(1)
