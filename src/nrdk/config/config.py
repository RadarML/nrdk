"""Configuration-related utilities to simplify hydra configurations."""

import os
from collections.abc import Mapping, Sequence
from typing import Any

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
