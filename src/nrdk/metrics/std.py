"""Batch-level standard deviation metric."""

import torch
import torch.distributed as dist
from jaxtyping import Float
from torch import Tensor


class BatchStd:
    """Population standard deviation across the batch axis.

    Computes the per-feature standard deviation of a `(batch, ch)` tensor
    across the batch dimension. In a distributed setting the statistic is
    aggregated across all ranks via a single `all_gather`, so the result
    reflects the full global batch on every rank.

    !!! note

        We use total variance to avoid catastrophic cancellation if naively
        accumulating moments.

    !!! warning

        This metric returns shape `(ch,)` — one value per feature dimension —
        rather than the `(batch,)` shape expected by the standard per-sample
        metric interface. It cannot be used directly in a metrics dict without
        further reduction.
    """

    def __call__(
        self, x: Float[Tensor, "batch ch"],
    ) -> Float[Tensor, "ch"]:
        """Compute batch population std.

        Args:
            x: input tensor.

        Returns:
            Population std per feature.
        """
        n, ch = x.shape
        total = x.sum(dim=0, dtype=torch.float64)
        # An empty local batch contributes nothing; a zero mean keeps its
        # (zero-weighted) between-rank term finite rather than NaN.
        mean = total / n if n > 0 else total
        stats = torch.stack([
            x.new_full((ch,), n, dtype=torch.float64), mean,
            ((x.double() - mean) ** 2).sum(dim=0)])  # [3, ch]

        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            flat = stats.new_empty((world_size * stats.shape[0], ch))
            dist.all_gather_into_tensor(flat, stats)
            stats = flat.view(world_size, *stats.shape)
        else:
            stats = stats[None]

        n_i, mean_i, m2_i = stats[:, 0], stats[:, 1], stats[:, 2]
        N = n_i.sum(dim=0)
        mean_global = (n_i * mean_i).sum(dim=0) / N
        # E[Var(x | rank)] + Var(E[x | rank]), as unnormalized second moments.
        m2 = m2_i.sum(dim=0) + (n_i * (mean_i - mean_global) ** 2).sum(dim=0)
        return (m2 / N).clamp(min=0).sqrt().to(x.dtype)
