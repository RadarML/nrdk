"""Meta-tests: confirm that the testing infrastructure is working."""

import pytest
import torch
from jaxtyping import TypeCheckError

from nrdk.metrics.metrics import lp_power, mean_with_mask


def test_builtin_type_hint_rejects_invalid_type():
    """`lp_power`'s `ord: int | float` rejects a non-numeric string."""
    x = torch.randn(4)
    with pytest.raises(TypeCheckError):
        lp_power(x, ord="two")  # type: ignore


def test_jaxtyping_array_hint_rejects_invalid_dtype():
    """`mean_with_mask`'s `Float[Tensor, ...]` rejects an integer tensor."""
    x = torch.ones(2, 3, dtype=torch.int64)
    with pytest.raises(TypeCheckError):
        mean_with_mask(x, None)  # type: ignore
