"""Configuration utilities."""

from jaxtyping import install_import_hook

with install_import_hook("nrdk.config", "beartype.beartype"):
    from .config import Singleton, expand

__all__ = ["expand", "Singleton"]
