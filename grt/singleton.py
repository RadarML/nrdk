"""Singleton registry."""

from typing import Any


class _SingletonRegistry:
    """Register an object as a singleton, by name."""

    def __init__(self) -> None:
        self.singletons: dict[str, Any] = {}

    def register(self, **kwargs) -> dict[str, Any]:
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
      _target_: grt.framework.Singleton.register
      example:
        _target_: path.to.module.ExampleClass
    ```

    Then, to use the registered `backend` later:
    ```yaml
    ...
      example:
        _target_: grt.framework.Singleton
        name: example
    ```
"""
