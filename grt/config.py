import os
from collections.abc import Mapping, Sequence

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

        if base is not None:
            _join = lambda p: os.path.join(base, p)
        else:
            _join = lambda p: p

        if isinstance(nested, Sequence):
            return [_join(item) for item in nested]
        elif isinstance(nested, Mapping):
            return sum((_expand(v, _join(k)) for k, v in nested.items()), [])
        else:
            raise ValueError("Nested structure must be a sequence or mapping.")

    return _expand(nested, base=path)
