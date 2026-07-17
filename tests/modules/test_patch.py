"""Tests for `nrdk.modules.patch`."""

import pytest
import torch

from nrdk.modules.patch import PatchMerge, Squeeze, Unpatch


def _small_input(shape, seed=0):
    """Generate a small random tensor with the given shape."""
    torch.manual_seed(seed)
    return torch.randn(*shape)


# PatchMerge


def test_patch_merge_even_shape():
    """An evenly-divisible input is merged and projected as expected."""
    x = _small_input((2, 8, 8, 4))
    out = PatchMerge(d_in=4, d_out=6, scale=[2, 2], norm=True)(x)

    assert out.shape == (2, 4, 4, 6)
    assert torch.all(torch.isfinite(out))


def test_patch_merge_crop_remainder():
    """`remainder="crop"` truncates the trailing, non-divisible elements."""
    x = _small_input((1, 5, 5, 4))
    out = PatchMerge(
        d_in=4, d_out=6, scale=[2, 2], norm=False, remainder="crop")(x)

    # 5 // 2 = 2 patches per axis after cropping the trailing element.
    assert out.shape == (1, 2, 2, 6)


def test_patch_merge_pad_remainder():
    """`remainder="pad"` zero-pads before merging non-divisible axes."""
    x = _small_input((1, 5, 5, 4))
    out = PatchMerge(
        d_in=4, d_out=6, scale=[2, 2], norm=False, remainder="pad")(x)

    # Padding adds 1 element per axis (5 -> 6), giving 3 patches per axis.
    assert out.shape == (1, 3, 3, 6)


@pytest.mark.parametrize("size,scale,expected_patches", [
    (5, 2, 3),   # remainder=1, pad=1 -> 6 -> 3 patches (scale=2 sanity check)
    (8, 4, 2),   # remainder=0, pad=0 -> 8 -> 2 patches (no spurious padding)
    (7, 3, 3),   # remainder=1, pad=2 -> 9 -> 3 patches
    (7, 4, 2),   # remainder=3, pad=1 -> 8 -> 2 patches
    (10, 4, 3),  # remainder=2, pad=2 -> 12 -> 3 patches
])
def test_patch_merge_pad_remainder_rounds_up_to_next_multiple(
    size, scale, expected_patches,
):
    """`remainder="pad"` pads up to the next multiple of `scale`.

    Regression test: padding by the raw remainder only coincidentally
    produces a divisible size when `scale=2`; other scale factors need
    `scale - remainder` (mod `scale`) to actually round up correctly.
    """
    x = _small_input((1, size, size, 4))
    out = PatchMerge(
        d_in=4, d_out=6, scale=[scale, scale], norm=False,
        remainder="pad")(x)

    assert out.shape == (1, expected_patches, expected_patches, 6)


def test_patch_merge_crop_matches_manual_crop():
    """Cropping should behave identically to pre-cropping the input."""
    x = _small_input((1, 5, 5, 4))
    cropped = x[:, :4, :4, :]

    merge = PatchMerge(d_in=4, d_out=6, scale=[2, 2], norm=False)
    out_full = merge(x)
    out_cropped = merge(cropped)

    assert torch.allclose(out_full, out_cropped)


@pytest.mark.parametrize("norm", [True, False])
def test_patch_merge_norm_toggle(norm):
    """The `norm` flag controls whether a `LayerNorm` module is created."""
    merge = PatchMerge(d_in=4, d_out=6, scale=[2, 2], norm=norm)
    assert (merge.norm is not None) == norm


def test_patch_merge_norm_changes_output():
    """Enabling `norm` changes the merged output relative to no norm."""
    x = _small_input((2, 8, 8, 4))

    torch.manual_seed(3)
    merge_norm = PatchMerge(d_in=4, d_out=6, scale=[2, 2], norm=True)
    torch.manual_seed(3)
    merge_nonorm = PatchMerge(d_in=4, d_out=6, scale=[2, 2], norm=False)

    out_norm = merge_norm(x)
    out_nonorm = merge_nonorm(x)
    assert not torch.allclose(out_norm, out_nonorm)


# Unpatch


def test_unpatch_2d_shape():
    """2D unpatching restores `(n, h, w, c)`."""
    unpatch = Unpatch(output_size=(8, 8, 3), features=16, size=(2, 2))
    x = torch.randn(2, 16, 16)  # xin = (8 // 2) * (8 // 2)

    out = unpatch(x)
    assert out.shape == (2, 8, 8, 3)


def test_unpatch_3d_shape():
    """3D unpatching restores `(n, x, y, z, c)`."""
    unpatch = Unpatch(output_size=(4, 4, 4, 2), features=10, size=(2, 2, 2))
    x = torch.randn(2, 8, 10)  # xin = 2 * 2 * 2

    out = unpatch(x)
    assert out.shape == (2, 4, 4, 4, 2)


def test_unpatch_4d_shape():
    """4D unpatching restores `(n, w, x, y, z, c)`."""
    unpatch = Unpatch(
        output_size=(4, 4, 4, 4, 1), features=6, size=(2, 2, 2, 2))
    x = torch.randn(2, 16, 6)  # xin = 2 * 2 * 2 * 2

    out = unpatch(x)
    assert out.shape == (2, 4, 4, 4, 4, 1)


def test_unpatch_unsupported_ndim_raises():
    """Unpatch only supports 2/3/4 spatial dimensions."""
    with pytest.raises(ValueError):
        Unpatch(output_size=(8, 3), features=6, size=(2,))(
            torch.randn(2, 4, 6))

    with pytest.raises(ValueError):
        Unpatch(
            output_size=(2, 2, 2, 2, 2, 3), features=6, size=(2, 2, 2, 2, 2)
        )(torch.randn(2, 1, 6))


def test_patch_merge_unpatch_roundtrip_shape():
    """PatchMerge followed by Unpatch recovers the original spatial shape."""
    h, w, d_in, d_scale, d_out, c_final = 8, 8, 4, 2, 6, 5
    x = _small_input((1, h, w, d_in))

    merge = PatchMerge(d_in=d_in, d_out=d_out, scale=[d_scale, d_scale])
    merged = merge(x)
    assert merged.shape == (1, h // d_scale, w // d_scale, d_out)

    flat = merged.reshape(1, -1, d_out)
    unpatch = Unpatch(
        output_size=(h, w, c_final), features=d_out,
        size=(d_scale, d_scale))
    restored = unpatch(flat)

    assert restored.shape == (1, h, w, c_final)


# Squeeze


def test_squeeze_n_channels_property():
    """`n_channels` multiplies the original sizes of the squeezed axes."""
    squeeze = Squeeze(dim=[0, 1], size=[2, 3, 5])
    assert squeeze.n_channels == 2 * 3


def test_squeeze_n_channels_without_size_raises():
    """`n_channels` requires `size` to be provided."""
    squeeze = Squeeze(dim=[0])
    with pytest.raises(ValueError):
        squeeze.n_channels


def test_squeeze_moves_axis_values_to_channels():
    """Squeezing axis 0 moves each element to a hand-traceable channel index.

    With `data[0, i, j, 0] = i * 3 + j` for a `(1, 2, 3, 1)` tensor, squeezing
    spatial axis `0` (size 2) into the channel axis should produce
    `out[0, j, i] == i * 3 + j`.
    """
    data = torch.zeros(1, 2, 3, 1)
    for i in range(2):
        for j in range(3):
            data[0, i, j, 0] = i * 3 + j

    squeeze = Squeeze(dim=[0], size=[2, 3])
    out = squeeze(data)

    assert out.shape == (1, 3, 2)
    for i in range(2):
        for j in range(3):
            assert out[0, j, i].item() == i * 3 + j


def test_squeeze_multiple_axes_channel_count():
    """Squeezing two spatial axes folds both sizes into the channel axis.

    The resulting channel count is the original channel size times the sizes
    of the squeezed axes; `n_channels` itself only reports the product of the
    squeezed axes' sizes (excluding the original channel dimension).
    """
    data = torch.randn(2, 3, 4, 5)  # n, d0, d1, c
    squeeze = Squeeze(dim=[0, 1], size=[3, 4, 5])

    out = squeeze(data)
    assert out.shape == (2, 3 * 4 * 5)
    assert squeeze.n_channels == 3 * 4
