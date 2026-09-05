"""Radar spectrum visualization utilities."""

import numpy as np
import torch
from abstract_dataloader.ext.objective import VisualizationConfig
from einops import rearrange, reduce
from jaxtyping import Float, Shaped
from torch import Tensor
from torch.nn.functional import interpolate

from .utils import tile_images


def _resize_float(
    image: Float[Tensor, "b h w"],
    vis_config: VisualizationConfig,
    mode: str = "nearest",
) -> Float[Tensor, "b h2 w2"]:
    kwargs = {} if mode == "nearest" else {"align_corners": False}
    return interpolate(
        image.unsqueeze(1), size=(vis_config.height, vis_config.width),
        mode=mode, **kwargs,
    ).squeeze(1)


def range_doppler(
    y_true: Float[Tensor, "b d el az rng ch"],
    y_pred: Float[Tensor, "b d el az rng ch"],
    vis_config: VisualizationConfig,
    eps: float = 0.0,
    include_phase: bool = True,
) -> dict[str, Shaped[np.ndarray, "H W 3"]]:
    """Render range-Doppler magnitude, log-magnitude, and phase visualizations.

    Args:
        y_true: ground-truth spectrum (single time slice), shape
            `(b, doppler, el, az, rng, ch)`.
        y_pred: predicted spectrum, same shape as `y_true`.
        vis_config: visualization configuration (cols, cmaps, ...).
        eps: if `>0`, clamp amplitudes to `[eps, inf)` before taking log.
        include_phase: whether to include a `"phase"` entry.  Requires
            at least 2 channels (complex data).

    Returns:
        Dict with keys `"magnitude"`, `"log_magnitude"`, and (if
            `include_phase`) `"phase"`, each an `H x W x 3` uint8 numpy array.
    """
    rd_true = reduce(y_true**2, "b d el az rng ch -> b rng d", "mean") ** 0.5
    rd_pred = reduce(y_pred**2, "b d el az rng ch -> b rng d", "mean") ** 0.5

    rd_true = _resize_float(rd_true, vis_config)
    rd_pred = _resize_float(rd_pred, vis_config)

    magnitude = tile_images(
        rd_true**0.5, rd_pred**0.5, torch.abs(rd_true**0.5 - rd_pred**0.5),
        cols=vis_config.cols,
        cmap=vis_config.cmaps.get("rd", "viridis"),
        normalize=True)

    if eps > 0.0:
        log_true = torch.log(rd_true.clamp(min=eps))
        log_pred = torch.log(rd_pred.clamp(min=eps))
    else:
        log_true = torch.log(rd_true)
        log_pred = torch.log(rd_pred)

    log_magnitude = tile_images(
        log_true, log_pred, torch.abs(log_true - log_pred),
        cols=vis_config.cols,
        cmap=vis_config.cmaps.get("rd_log", "viridis"),
        normalize=True)

    out: dict[str, Shaped[np.ndarray, "H W 3"]] = {
        "magnitude": magnitude,
        "log_magnitude": log_magnitude,
    }

    if include_phase:
        p_true = rearrange(
            torch.atan2(y_true[:, :, 0, 0, :, 1], y_true[:, :, 0, 0, :, 0]),
            "b d rng -> b rng d")
        p_pred = rearrange(
            torch.atan2(y_pred[:, :, 0, 0, :, 1], y_pred[:, :, 0, 0, :, 0]),
            "b d rng -> b rng d")

        p_true = _resize_float(p_true, vis_config, mode="nearest")
        p_pred = _resize_float(p_pred, vis_config, mode="nearest")

        p_true_n = p_true / (2 * np.pi) + 0.5
        p_pred_n = p_pred / (2 * np.pi) + 0.5
        phase_diff = ((p_pred_n - p_true_n) + 0.5) % 1.0

        out["phase"] = tile_images(
            p_true_n, p_pred_n, phase_diff,
            cols=vis_config.cols,
            cmap=vis_config.cmaps.get("phase", "coolwarm"),
            normalize=False)

    return out


def range_azimuth(
    y_true: Float[Tensor, "b d el az rng ch"],
    y_pred: Float[Tensor, "b d el az rng ch"],
    vis_config: VisualizationConfig,
) -> dict[str, Shaped[np.ndarray, "H W 3"]]:
    """Render range-azimuth amplitude visualizations.

    The amplitude is averaged over Doppler and elevation dimensions and then
    resized to `vis_config.height x vis_config.width` via bilinear
    interpolation.

    Args:
        y_true: ground-truth spectrum (single time slice), shape
            `(b, doppler, el, az, rng, ch)`.
        y_pred: predicted spectrum, same shape as `y_true`.
        vis_config: visualization configuration (cols, cmaps, height, width).

    Returns:
        Dict with key `"range_azimuth"`, an `H x W x 3` uint8 numpy array.
    """
    ra_true = reduce(y_true**2, "b d el az rng ch -> b rng az", "mean") ** 0.5
    ra_pred = reduce(y_pred**2, "b d el az rng ch -> b rng az", "mean") ** 0.5

    ra_true = _resize_float(ra_true, vis_config)
    ra_pred = _resize_float(ra_pred, vis_config)

    return {"range_azimuth": tile_images(
        ra_true**0.5, ra_pred**0.5, torch.abs(ra_true**0.5 - ra_pred**0.5),
        cols=vis_config.cols,
        cmap=vis_config.cmaps.get("ra", "viridis"),
        normalize=True)}
