"""Configuration-related utilities to simplify hydra configurations."""

from nrdk._typecheck import typechecker

with typechecker("nrdk.config"):
    from .batch_size import autoscale_batch_size
    from .config import PreventHydraOverwrite, Singleton, expand, inst_from
    from .logging import configure_rich_logging

__all__ = [
    "expand", "inst_from", "Singleton", "PreventHydraOverwrite",
    "configure_rich_logging", "autoscale_batch_size"
]
