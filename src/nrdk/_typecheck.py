"""Type checking hooks."""

import os
from contextlib import AbstractContextManager, nullcontext

from jaxtyping import install_import_hook


def typechecker(mod_name: str) -> AbstractContextManager:
    """Type checking entry point."""
    if os.environ.get("JAXTYPING_DISABLE", "").lower() in ("1", "true", "yes"):
        return nullcontext()
    else:
        return install_import_hook(mod_name, "beartype.beartype")
