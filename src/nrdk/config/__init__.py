"""Configuration-related utilities to simplify hydra configurations."""

from jaxtyping import install_import_hook

with install_import_hook("nrdk.config", "beartype.beartype"):
    from .config import Singleton, expand, inst_from

__all__ = ["expand", "inst_from", "Singleton"]
