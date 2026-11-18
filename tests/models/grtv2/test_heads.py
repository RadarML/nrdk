"""Tests for GRTv2 heads."""

import pytest
import torch

from nrdk.models.grtv2 import LinearHead

D_MODEL = 16


@pytest.mark.parametrize("squeeze", [True, False])
def test_linear_head_output_shape(squeeze):
    """Patches are reassembled into the full output grid."""
    head = LinearHead(
        d_model=D_MODEL, out_channels=1, output_size=[1, 8, 4],
        patch_size=[1, 4, 2], squeeze=squeeze)
    out = head(torch.randn(2, 1 * 2 * 2, D_MODEL))
    assert out.shape == ((2, 1, 8, 4) if squeeze else (2, 1, 8, 4, 1))


def test_linear_head_int_patch_size():
    """An int patch size is applied to every axis."""
    head = LinearHead(
        d_model=D_MODEL, out_channels=3, output_size=[4, 4], patch_size=2)
    assert head(torch.randn(2, 4, D_MODEL)).shape == (2, 4, 4, 3)
