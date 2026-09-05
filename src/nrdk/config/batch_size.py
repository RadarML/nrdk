"""Batch size auto-scaling."""

import logging

import torch

logger = logging.getLogger(__name__)


def autoscale_batch_size(batch: int, accumulation: int = 1) -> int:
    """Scale a configured batch size down to a per-step, per-device size.

    The provided `batch` size is treated as the target *effective* batch
    size, i.e., after gradient accumulation and across all GPUs. This is
    divided down by the accumulation factor and then by the number of
    GPUs (via `torch.cuda.device_count()`) to get the batch size that should be
    used by each dataloader step on each device.

    !!! warning

        If `accumulation * n_gpus` does not evenly divide the batch size, the
        logical batch size will be implicitly rounded down to the nearest
        multiple.

    Args:
        batch: target effective batch size.
        accumulation: number of gradient accumulation steps.

    Returns:
        Per-step, per-device batch size.

    Raises:
        ValueError: if the batch size is scaled down to zero, i.e. the target
            effective batch size is smaller than the total scaling factor.
    """
    if accumulation < 1:
        raise ValueError(
            f"Accumulation factor must be >= 1, got {accumulation}.")

    if accumulation > 1:
        logger.info(
            f"Auto-scaling batch size by accumulation={accumulation}: "
            f"{batch} -> {batch // accumulation}")
        if batch < accumulation:
            raise ValueError(
                f"Batch size {batch} is smaller than "
                f"accumulation={accumulation}, which would scale it down to "
                "zero.")
        if batch % accumulation != 0:
            logger.warning(
                f"Batch size {batch} is not divisible by "
                f"accumulation={accumulation}.")

        batch = batch // accumulation

    n_gpus = torch.cuda.device_count()
    if n_gpus > 1:
        logger.info(
            f"Auto-scaling batch size by n_gpus={n_gpus}: "
            f"{batch} -> {batch // n_gpus}")
        if batch < n_gpus:
            raise ValueError(
                f"Post-accumulation (={accumulation}) batch size {batch} is "
                f"smaller than n_gpus={n_gpus}, which would scale it down to "
                "zero.")
        if batch % n_gpus != 0:
            logger.warning(
                f"Batch size {batch} is not divisible by n_gpus={n_gpus}.")

        batch = batch // n_gpus

    return batch
