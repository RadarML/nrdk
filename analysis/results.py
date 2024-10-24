"""Experiments management."""

import os
import re
from functools import cache
from multiprocessing import Pool

import numpy as np
from beartype.typing import Iterator, Optional
from jaxtyping import Float, Shaped

from analysis.stats import NDStats, effective_sample_size

from .result import ComparativeStats, Result


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

    def __len__(self) -> int:
        return len(self.results)

    @cache
    def __getitem__(self, name: str) -> Result:
        return Result(os.path.join(self.path, name))

    def __iter__(self) -> Iterator[str]:
        """Dict keys-like iterator; alias for `.keys()`"""
        return self.keys()

    def keys(self) -> Iterator[str]:
        """Dict keys-like iterator."""
        return iter(self.results)

    def items(self) -> Iterator[tuple[str, Result]]:
        """Dict key/value-like iterator."""
        return ((k, self[k]) for k in self)

    def values(self) -> Iterator[Result]:
        """Dict values-like iterator."""
        return (self[k] for k in self)

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
