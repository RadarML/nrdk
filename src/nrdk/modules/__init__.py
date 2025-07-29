"""General purpose pytorch modules with n-dimensional support where applicable.

These modules are designed to act as an interface to facilitate model training
and transfer learning; however, **when in doubt, users should simply default to
implementing and maintaining their own copies of each module**. Only
generic and reusable modifications should be made here, with dataset,
objective, and method-specific implementations or modifications made in
downstream repositories.
"""

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
