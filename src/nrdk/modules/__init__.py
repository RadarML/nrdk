"""General purpose modules with n-dimensional support."""

from jaxtyping import install_import_hook

with install_import_hook("nrdk.modules", "beartype.beartype"):
    from .patch import PatchMerge, Squeeze, Unpatch
    from .position import BasisChange, LearnableND, Readout, Sinusoid

__all__ = [
    "PatchMerge", "Unpatch", "Squeeze",
    "Sinusoid", "LearnableND", "Readout", "BasisChange"
]
