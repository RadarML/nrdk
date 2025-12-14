"""Configuration-related utilities to simplify hydra configurations."""

from jaxtyping import install_import_hook

with install_import_hook("nrdk.config", "beartype.beartype"):
    from .config import PreventHydraOverwrite, Singleton, expand, inst_from
    from .logging import configure_rich_logging

__all__ = [
    "expand", "inst_from", "Singleton", "PreventHydraOverwrite",
    "configure_rich_logging"
]
