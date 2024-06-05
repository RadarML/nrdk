"""Base classes."""

from beartype.typing import Any


class BaseTransform:
    """Generic transformation.

    args:
        path: dataset path to load any required metadata.
    """

    def __init__(self, path: str) -> None:
        pass

    def __call__(self, data: Any, aug: dict[str, Any] = {}) -> Any:
        raise NotImplementedError()
