"""Batch-level standard deviation metric."""

import torch
import torch.distributed as dist
from jaxtyping import Float
from torch import Tensor


class BatchStd:
    """Population standard deviation across the batch axis.

    Computes the per-feature standard deviation of a `(batch, ch)` tensor
    across the batch dimension. In a distributed setting the statistic is
    aggregated across all ranks via a single `all_reduce`, so the result
    reflects the full global batch on every rank.

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
        B = x.shape[0]
        stats = torch.stack([x.sum(dim=0), (x ** 2).sum(dim=0)])  # [2, ch]
        N = float(B)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(stats)
            N *= dist.get_world_size()
        var = stats[1] / N - (stats[0] / N) ** 2
        return var.clamp(min=0).sqrt()
