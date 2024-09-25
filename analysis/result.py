"""Results metadata."""

import os
import re
from functools import cache
from multiprocessing import Pool

import numpy as np
from beartype.typing import NamedTuple, Optional, cast
from jaxtyping import Float, Shaped
from scipy.stats import norm

from analysis.stats import NDStats, effective_sample_size


class ComparativeStats(NamedTuple):
    """Statistics comparing a set of methods.

    Attributes:
        traces: list of traces represented by this data.
        abs: stats relating to the absolute value of the metric in question.
        diff: stats relating to the relative value of the metric in question.
    """

    traces: list[str]
    abs: NDStats
    diff: NDStats

    @staticmethod
    def stack(*stats) -> "ComparativeStats":
        """Stack multiple statistics containers."""
        return ComparativeStats(
            traces=sum((s.traces for s in stats), start=[]),
            abs=NDStats.stack(*[s.abs for s in stats]),
            diff=NDStats.stack(*[s.diff for s in stats]))

    def sum(self) -> "ComparativeStats":
        """Get aggregate values.

        Note that while `abs` and `diff` will become aggregated statistics
        (and lose their leading vector dimension), `traces` remains the same
        length, and still lists all source trace names.
        """
        return ComparativeStats(
            traces=self.traces, abs=self.abs.sum(), diff=self.diff.sum())

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


class Result:
    """Result/evaluation container."""

    def __init__(self, path: str) -> None:
        if not os.path.exists(path):
            raise ValueError(
                f"{path} does not appear to be a valid results folder.")
        self.path = path
        self.eval = self._discover_evaluations(path)

    def _discover_evaluations(self, path: str) -> list[str]:
        """Discover evaluations in a result directory."""
        manifest = []
        base = os.path.join(path, "eval")
        for root, _, files in os.walk(base):
            for file in files:
                manifest.append(
                    os.path.relpath(os.path.join(root, file), base))
        return manifest

    def __getitem__(self, eval: str):
        """Load a trace (as a NpzFile).

        NOTE: don't cache here to release the loaded NpzFile. The OS should
        handle caching here if we hit the same file repeatedly.
        """
        return np.load(os.path.join(self.path, "eval", eval))


class Results:
    """Results container.

    Args:
        path: path to the root results folder. `path` itself can be a symlink,
            but symlinks inside `path` are not followed.
        marker_name: filename to search for; each directory inside `path`
            containing a file with this name is considered a single "result".
    """

    def __init__(
        self, path: str = "results", marker_name: str = "hparams.yaml"
    ) -> None:
        self.path = path
        self.marker_name = marker_name
        self.results = sorted(self._discover_results(path))

    @cache
    def __getitem__(self, name: str) -> Result:
        return Result(os.path.join(self.path, name))

    def _discover_results(self, path: str) -> list[str]:
        """Discover `deepradar` results in a root directory.

        NOTE: This method does not follow symlinks (to prevent infinite loops).
        """
        manifest = []
        for root, _, files in os.walk(path):
            if self.marker_name in files:
                manifest.append(os.path.relpath(root, path))
        return manifest

    def compare(
        self, results: Optional[list[str]] = None, key: str = "loss",
        pattern: Optional[str] = None
    ) -> ComparativeStats:
        """Get comparison statistics between a list of experiments.

        Args:
            results: list of experiments to compare. Should be file paths,
                relative to the base path of this container.
            key: metric name to compare on.
            pattern: regex filter to apply to traces.

        Returns:
            Comparison statistics between the specified experiments. Only
            evaluation traces which are present in all of the specified
            experiments are used.

        Raises:
            ValueError: An invalid configuration is specified:
                - if the specified experiments have no evaluation commonality.
                - if the specified list of results contain duplicates.
        """
        if results is None:
            results = self.results

        if len(results) != len(set(results)):
            raise ValueError(f"Cannot compare duplicate methods: {results}.")

        i, j = np.triu_indices(len(results), 1)

        def unflatten(
            vec: Float[np.ndarray, "C"], signed: bool = False
        ) -> Float[np.ndarray, "Nr Nr"]:
            """Expand `0 <= i < j < Nr` indices to a square matrix."""
            mat = np.zeros((len(results), len(results)))
            mat[i, j] = vec
            mat[j, i] = -vec if signed else vec
            return mat

        def stats(trace: str) -> ComparativeStats:
            """Calculate statistics."""
            arr: Shaped[np.ndarray, "Nr N"] = np.stack([
                self[r][trace][key] for r in results])

            with Pool(processes=arr.shape[0]) as p:
                mean_ess = np.array(p.map(effective_sample_size, arr))

            diff_raw = arr[i] - arr[j]
            with Pool(processes=diff_raw.shape[0]) as p:
                diff_ess_flat = p.map(effective_sample_size, diff_raw)

            return ComparativeStats(
                traces=[trace],
                abs=NDStats(
                    n=np.array(arr.shape[1]),
                    m1=np.sum(arr, axis=1),
                    m2=np.sum(arr**2, axis=1),
                    ess=mean_ess),
                diff=NDStats(
                    n=np.array(arr.shape[1]),
                    m1=unflatten(np.sum(diff_raw, axis=1), signed=True),
                    m2=unflatten(np.sum(diff_raw**2, axis=1)),
                    ess=unflatten(np.array(diff_ess_flat))))

        common = sorted(list(
            set.intersection(*[set(self[r].eval) for r in results])))
        if len(common) == 0:
            raise ValueError(
                "The specified results do not have any evaluation traces in "
                "common.")

        if pattern is not None:
            common = [c for c in common if re.match(pattern, c)]
            if len(common) == 0:
                raise ValueError(
                    f"No traces common matched the filter pattern {pattern}.")

        return ComparativeStats.stack(*[stats(t) for t in common])
