"""Experiment comparisons."""

import numpy as np
from beartype.typing import NamedTuple, Self, cast
from jaxtyping import Float
from scipy.stats import norm

from analysis.stats import NDStats


class BaselineStats(NamedTuple):
    """Statistics comparing a set of methods against a fixed baseline.

    Attributes:
        traces: list of traces represented by this data.
        abs: stats relating to the absolute value of the metric in question.
        diff: stats relating to the relative value of the metric in question.
    """

    traces: list[str]
    abs: NDStats
    diff: NDStats

    @classmethod
    def stack(cls, *stats) -> Self:
        """Stack multiple statistics containers."""
        return cls(
            traces=sum((s.traces for s in stats), start=[]),
            abs=NDStats.stack(*[s.abs for s in stats]),
            diff=NDStats.stack(*[s.diff for s in stats]))


    def sum(self) -> Self:
        """Get aggregate values.

        Note that while `abs` and `diff` will become aggregated statistics
        (and lose their leading vector dimension), `traces` remains the same
        length, and still lists all source trace names.
        """
        return self.__class__(
            traces=self.traces, abs=self.abs.sum(), diff=self.diff.sum())


class ComparativeStats(BaselineStats):
    """Statistics comparing each unique pair within a set of methods.

    Attributes:
        traces: list of traces represented by this data.
        abs: stats relating to the absolute value of the metric in question.
        diff: stats relating to the relative value of the metric in question.
    """

    def z_boundary(
        self, p: float = 0.05, corrected: bool = False, subgroups: int = 1
    ) -> float:
        """Get z-score cutoff; see :py:meth:`.significance`."""
        p_target = p / 2
        if corrected:
            n_methods = self.abs.m1.shape[-1]
            n_inferences = n_methods * (n_methods - 1) / 2 * subgroups
            p_target = p_target / n_inferences

        # norm.ppf returns a np.float64 (float subclass) with scalar input.
        return cast(float, norm.ppf(1 - p_target))

    def significance(
        self, p: float = 0.05, corrected: bool = False, subgroups: int = 1
    ) -> Float[np.ndarray, "*batch Nr Nr"]:
        """Get significance matrix of the differences.

        Args:
            p: target p-value for a 2-sided test (actually one-sided tests for
                either end with a union bound).
            corrected: whether to correct for multiple inference using the
                bonferroni (union) correction.
            subgroups: number of subgroups being compared in total (i.e. the
                number of parallel `ComparativeStats` in play) to perform
                multiple inference over all hypotheses, not just those
                contained here.

        Returns:
            Significance matrix, where +1 indicates that each row value is
            significantly greater than the column value, -1 indicates
            significantly less, and 0 indicates no significant difference.
        """
        boundary = self.z_boundary(
            p=p, corrected=corrected, subgroups=subgroups)
        return (
            1.0 * (self.diff.zscore > boundary)
            - 1.0 * (self.diff.zscore < -boundary))

    def percent(self) -> Float[np.ndarray, "*batch Nr Nr"]:
        """Get percent difference, relative to each column.

        Read as "(row index) is x% more/less than (column index)".
        """
        return 100. * self.diff.mean / self.abs.mean[..., None, :]
