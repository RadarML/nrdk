"""Windowed attention modules."""

import numpy as np
import torch
from beartype.typing import Sequence
from jaxtyping import Bool, Float, Integer, Shaped
from torch import Tensor, nn


class WindowPartition:
    """Partition a N-dimensional tensor into arbitrary-sized windows.

    Equivalent to a N-d implementation of `window_partition` and
    `window_reverse` in [M6]_.

    As a quick test if you wish to make modifications::

        wp = WindowPartition((3, 4, 6), (3, 2, 2))
        x = torch.rand(5, 3, 4, 6, 7)
        xp = wp.partition(x)
        xpp = wp.unpartition(xp)
        assert(np.allclose(x.numpy(), xpp.numpy()))

    Args:
        size: input tensor spatial dimensions (i.e. excluding batch, channels).
        window_size: window size for each of the spatial axes. Note that unlike
            [M6]_, each dimension can have a different window size.
    """

    def __init__(self, size: Sequence[int], window: Sequence[int]) -> None:
        assert len(size) == len(window)

        self.size = size
        self.window = window
        self.wsize = int(np.prod(window))
        self.n_windows = int(np.prod(size) / np.prod(window))

        nd = len(size)
        self.target_shape: list[int] = sum(
            ([ds // ws, ws] for ds, ws in zip(size, window)), start=[])
        self.target_order = (
            [0] + [2 * i + 1 for i in range(nd)]
            + [2 * i + 2 for i in range(nd)] + [nd * 2 + 1])
        self.reverse_shape: list[int] = (
            [ds // ws for ds, ws in zip(size, window)] + list(window))
        self.reverse_order = sum(
            [[i + 1, i + nd + 1] for i in range(nd)], start=[0]) + [nd * 2 + 1]

    def partition(
        self, x: Shaped[Tensor, "b *spatial c"]
    ) -> Shaped[Tensor, "b2 spatial2 c"]:
        """Partition into windows.

        Args:
            x: input tensor in batch-spatial-channel order. The spatial axes
                must match `WindowPartition.size`.

        Returns:
            Partitioned version of `x`; partitions are absorbed into the batch.
        """
        batch, *_, channels = x.shape
        return (
            x.reshape([batch] + self.target_shape + [channels])
            .permute(self.target_order).contiguous()
            .reshape(-1, int(np.prod(self.window)), channels))

    def unpartition(
        self, x: Shaped[Tensor, "b2 spatial2 c"]
    ) -> Shaped[Tensor, "b *spatial c"]:
        """Unpartition windows into a N-dimensional representation.

        Args:
            x: input tensor in batch-spatial-channel order, with different
                windows absorbed into the batch.

        Returns:
            "Unwindowed" embeddings.
        """
        channels = x.shape[-1]
        return (
            x.reshape(-1, *self.reverse_shape, channels)
            .permute(self.reverse_order).contiguous()
            .reshape(-1, *self.size, channels))

    def attention_mask(
        self, shift: Sequence[int]
    ) -> Bool[Tensor, "windows wsize wsize"]:
        """Create attention mask.

        For each nonzero axis in shift:

        - We interpret the shift as a leftward shift.
        - For the first window in that axis, we create a mask which blocks
          `:shift` from attending to `shift:`, and vice versa.
        - Note only the first window (index 0) needs to be blocked, since it
          becomes the same "logical" window as the last window once the axis is
          left shifted.

        Args:
            shift: shift sizes; must have the same length as `size` and
                `window`. If a attention masking is not desired for an axis
                `k`, pass `shift[k]=0` regardless of what the actual shift
                for that axis was.

        Returns:
            Attention mask. Has (flattened) axes
            `[*windows, *window_size, *window_size]`, where
            `[p_1, ... p_n, x_1, ... x_n, y_1, ... y_n]` corresponds to
            whether element `(x_1, ... x_n)` should be blocked from
            attending to `(y_1, ... y_n)` on window `(p_1, ... p_n)`.
        """
        assert len(shift) == len(self.size)

        def _onehot_slice(i, val: int | slice) -> list[int | slice]:
            slices: list[int | slice] = [slice(None)] * len(self.size)
            slices[i] = val
            return slices

        mask = torch.zeros(
            [s // w for s, w in zip(self.size, self.window)]
            + list(self.window) + list(self.window), dtype=torch.bool)

        # ... for each shift amount:
        for i, s in enumerate(shift):
            if s != 0:
                # The 'split' window is always left aligned.
                partition = _onehot_slice(i, 0)
                # left, right shift slices
                sl = _onehot_slice(i, slice(None, s))
                sr = _onehot_slice(i, slice(s, None))

                mask[partition + sl + sr] = True
                mask[partition + sr + sl] = True

        return mask.reshape(self.n_windows, self.wsize, self.wsize)


class RelativePositionBias(nn.Module):
    """N-dimensional positional bias encoding.

    For a window size `[*window]`, we learn `prod(window) * n_heads`
    different biases, where bias `[(x1 ... xn) h]` corresponds to attention
    between `[(i1 ... in) (j1 ... jn)]` such that `xk = |ik - jk|` for head
    `h`. Note that the bias is applied to a flattened array.

    Args:
        window: window size.
        n_head: number of attention heads; each head learns a different bias.
    """

    def __init__(self, window: Sequence[int], n_head: int = 10) -> None:
        super().__init__()

        self.wsize = int(np.prod(window))
        self.window = window

        indices = self._attention_bias_indices()
        self.register_buffer("indices", indices)
        self.bias = nn.Parameter(torch.zeros(self.wsize, n_head))

    def _attention_bias_indices(self) -> Integer[Tensor, "..."]:
        """Get attention bias indices for this window size."""
        Nd = len(self.window)
        pos_abs = torch.meshgrid([torch.arange(k) for k in self.window] * 2)
        pos_rel = [
            torch.abs(i - j) for i, j in zip(pos_abs[:Nd], pos_abs[Nd:])]
        rel_to_bias = torch.arange(
            self.wsize, dtype=torch.int).reshape(self.window)
        return rel_to_bias[pos_rel]

    def forward(self) -> Float[Tensor, "n_heads wsize wsize"]:
        """Get indexed relative position bias.

        Returns:
            Position biases between a `src` tensor attending to a `dst` tensor
            with `n_heads` different heads.
        """
        return self.bias[self.indices].reshape(
            self.wsize, self.wsize, -1).permute(2, 0, 1)
