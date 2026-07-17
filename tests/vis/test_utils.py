"""Tests for vis utilities (tile_images, swap_angular_conventions)."""

import numpy as np
import torch

from nrdk.vis import swap_angular_conventions, tile_images


def _small_rgb_images():
    """Generate small RGB test images (2,3,5,3)."""
    torch.manual_seed(42)
    return torch.randn(2, 3, 5, 3)


def _small_grayscale_images():
    """Generate small grayscale test images (2,3,5)."""
    torch.manual_seed(42)
    return torch.randn(2, 3, 5)


# tile_images


def test_tile_images_grayscale_basic():
    """Test tile_images with grayscale images."""
    images = _small_grayscale_images()  # (2,3,5)
    result = tile_images(images, cols=2, normalize=True)

    # Should create 1 row × 2 cols grid: (3, 10, 3)
    assert result.shape == (3, 10, 3)
    assert result.dtype == np.uint8


def test_tile_images_rgb_basic():
    """Test tile_images with RGB images."""
    images = _small_rgb_images()  # (2,3,5,3)
    result = tile_images(images, cols=2, normalize=True)

    # Should create 1 row × 2 cols grid: (3, 10, 3)
    assert result.shape == (3, 10, 3)
    assert result.dtype == np.uint8


def test_tile_images_multiple_predictions():
    """Test tile_images with y_true and y_hat."""
    y_true = _small_grayscale_images()  # (2,3,5)
    y_hat = _small_grayscale_images() * 0.5  # Different values

    result = tile_images(y_true, y_hat, cols=2, normalize=True)

    # Should stack vertically: y_true on top, y_hat below
    # Result: (6, 10, 3) - doubled height
    assert result.shape == (6, 10, 3)


def test_tile_images_padding():
    """Test tile_images with non-divisible batch size."""
    # Create 3 images but request 2 columns
    images = torch.randn(3, 3, 5)
    result = tile_images(images, cols=2, normalize=True)

    # Should pad to 4 images (2 rows × 2 cols): (6, 10, 3)
    assert result.shape == (6, 10, 3)


def test_tile_images_custom_colormap_str():
    """Test tile_images with a named matplotlib colormap."""
    images = _small_grayscale_images()  # (2,3,5)
    result = tile_images(images, cols=1, cmap='plasma', normalize=True)

    assert result.shape == (6, 5, 3)
    assert result.dtype == np.uint8


def test_tile_images_no_normalization():
    """Test tile_images without normalization."""
    # Pre-normalized data in [0, 1]
    images = torch.rand(2, 3, 5)
    result = tile_images(images, cols=2, normalize=False)

    assert result.shape == (3, 10, 3)


def test_tile_images_bool_data_normalize():
    """Test tile_images normalizes bool-typed images."""
    images = torch.zeros(2, 3, 5, dtype=torch.bool)
    images[0, 1, 2] = True
    result = tile_images(images, cols=2, normalize=True)

    assert result.shape == (3, 10, 3)
    assert result.dtype == np.uint8


def test_tile_images_single_image():
    """Test tile_images with single image."""
    image = torch.randn(1, 3, 5)
    result = tile_images(image, cols=1, normalize=True)

    assert result.shape == (3, 5, 3)


def test_tile_images_more_cols_than_images():
    """Test tile_images when cols > batch_size."""
    images = torch.randn(2, 3, 5)
    result = tile_images(images, cols=5, normalize=True)  # ncols > images
    assert result.shape == (3, 10, 3)  # 1 row × 2 actual images


# swap_angular_conventions


def test_swap_angular_conventions_basic():
    """Test basic swap_angular_conventions functionality."""
    data = torch.arange(24).reshape(2, 3, 4)
    result = swap_angular_conventions(data)
    assert result.shape == (2, 4, 3)

    # Verify it's not just a transpose (would miss the flip)
    simple_transpose = data.transpose(-1, -2)
    assert not torch.allclose(result, simple_transpose)


def test_swap_angular_conventions_channels():
    """Test swap_angular_conventions with channel dimension."""
    data = torch.randn(2, 3, 5, 2)
    result = swap_angular_conventions(data)
    assert result.shape == (2, 5, 3, 2)


def test_swap_angular_conventions_identity():
    """Test that applying swap twice returns original."""
    data = torch.randn(2, 3, 5)
    result = swap_angular_conventions(swap_angular_conventions(data))

    assert torch.allclose(result, data)
    assert result.shape == data.shape


def test_swap_angular_conventions_single_pixel():
    """Test swap_angular_conventions with minimal data."""
    data = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])  # (1,2,2)
    result = swap_angular_conventions(data)

    assert result.shape == (1, 2, 2)
    # The actual transformation: flip(dims=[1,2]) then transpose(1,2)
    # [[1,2], [3,4]] -> flip -> [[4,3], [2,1]] -> transpose -> [[4,2], [3,1]]
    expected = torch.tensor([[[4.0, 2.0], [3.0, 1.0]]])
    assert torch.allclose(result, expected)
