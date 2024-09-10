"""Results metadata."""

import os
from functools import cache
from multiprocessing import Pool

import numpy as np
from beartype.typing import NamedTuple, Optional, cast
from jaxtyping import Float, Shaped
from scipy.stats import norm

from analysis.stats import NDStats, effective_sample_size


class ComparativeStats(NamedTuple):
    """Statistics comparing a set of methods."""

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
        """Get aggregate values."""
        return ComparativeStats(
            traces=self.traces, abs=self.abs.sum(), diff=self.diff.sum())

    def z_boundary(self, p: float = 0.05, corrected: bool = False) -> float:
        """Get z-score cutoff; see :py:meth:`.significance`."""
        p_target = p / 2
        if corrected:
            n_methods = self.abs.m1.shape[-1]
            n_inferences = n_methods * (n_methods - 1) / 2
            p_target = p_target / n_inferences

        # norm.ppf returns a np.float64 (float subclass) with scalar input.
        return cast(float, norm.ppf(1 - p_target))

    def significance(
        self, p: float = 0.05, corrected: bool = False
    ) -> Float[np.ndarray, "*batch Nr Nr"]:
        """Get significance matrix of the differences.

        Args:
            p: target p-value for a 2-sided test (actually one-sided tests for
                either end with a union bound).
            corrected: whether to apply the bonferroni (union) correction.

        Returns:
            Significance matrix, where +1 indicates that each row value is
            significantly greater (worse) than the column value, -1 indicates
            significantly less (better), and 0 indicates no significance.
        """
        boundary = self.z_boundary(p=p, corrected=corrected)
        return (
            1.0 * (self.diff.zscore > boundary)
            - 1.0 * (self.diff.zscore < -boundary))


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
    """Results container."""

    def __init__(self, path: str) -> None:
        self.path = path
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
            if 'hparams.yaml' in files:
                manifest.append(os.path.relpath(root, path))
        return manifest

    def compare(
        self, results: Optional[list[str]] = None, key: str = "loss"
    ) -> ComparativeStats:
        if results is None:
            results = self.results
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
        return ComparativeStats.stack(*[stats(t) for t in common])
