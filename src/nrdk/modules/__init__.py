"""General purpose pytorch modules with n-dimensional support."""

from jaxtyping import install_import_hook

with install_import_hook("nrdk.modules", "beartype.beartype"):
    from .conv import ConvNextLayer
    from .patch import PatchMerge, Squeeze, Unpatch
    from .position import BasisChange, LearnableND, Readout, Sinusoid

__all__ = [
    "ConvNextLayer",
    "PatchMerge", "Unpatch", "Squeeze",
    "Sinusoid", "LearnableND", "Readout", "BasisChange"
]
