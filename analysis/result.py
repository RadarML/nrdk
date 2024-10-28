"""Experiment metadata."""

import os
import re

import numpy as np
from beartype.typing import Iterator, Mapping, NamedTuple, Optional, cast
from jaxtyping import Float, Shaped
from scipy.stats import norm
from tensorboard.backend.event_processing import event_accumulator

from analysis.stats import NDStats


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
                if file.endswith(".npz"):
                    manifest.append(
                        os.path.relpath(os.path.join(root, file), base))
        return manifest

    def __getitem__(self, eval: str) -> Mapping[str, np.ndarray]:
        """Load a trace (as a NpzFile).

        NOTE: don't cache here to release the loaded NpzFile. The OS should
        handle caching here if we hit the same file repeatedly.
        """
        return np.load(os.path.join(self.path, "eval", eval))

    def __iter__(self) -> Iterator[str]:
        """Dict keys-like iterator; alias for `.keys()`"""
        return self.keys()

    def keys(self) -> Iterator[str]:
        """Dict keys-like iterator."""
        return iter(self.eval)

    def items(self) -> Iterator[tuple[str, Mapping[str, np.ndarray]]]:
        """Dict key/value-like iterator."""
        return ((k, self[k]) for k in self)

    def values(self) -> Iterator[Mapping[str, np.ndarray]]:
        """Dict values-like iterator."""
        return (self[k] for k in self)

    def load_all(self, pattern: Optional[str] = None):
        """Load all traces (as NpzFiles).

        If a regex `pattern` is specified, only traces which match the pattern
        are loaded.
        """
        traces = self.eval
        if pattern is not None:
            traces = [c for c in traces if re.match(pattern, c)]
            if len(traces) == 0:
                raise ValueError(
                    f"No traces common matched the filter pattern {pattern}.")

        return [self[t] for t in traces]

    def summarize_training(
        self, create: bool = False
    ) -> Mapping[str, Shaped[np.ndarray, "..."]]:
        """Get training summary.

        This method expects to find a single `*events.out.tfevents*` in the
        result directory. All scalar events are extracted and written to a
        single `meta.npz` file.

        Args:
            create: whether to create the `meta.npz` summary if not present,
                or update if the `tfevents` file is newer.

        Returns:
            `npz` file, if `meta.npz` already exists or is up to date, or
            newly created summary.

        Raises:
            FileNotFoundError: missing file, e.g. events file, `meta.json` if
                `create=False`.
            ValueError: no events file or too many events files.
        """
        meta_path = os.path.join(self.path, "meta.npz")

        # Early exit 1: creating is not allowed
        if not create:
            if not os.path.exists(meta_path):
                raise FileNotFoundError(
                    "Summarized metadata `meta.npz` does not exist, and "
                    "`create=False`.")
            else:
                return np.load(meta_path)

        events_candidates = [
            x for x in os.listdir(self.path)
            if x.startswith("events.out.tfevents")]
        if len(events_candidates) > 1:
            raise ValueError(f"More than one events file: {events_candidates}")
        if len(events_candidates) == 0:
            raise ValueError("No events files found.")
        events_file = os.path.join(self.path, events_candidates[0])

        # Early exit 2: file exists and is up to date
        t_event = os.path.getmtime(events_file)
        t_meta = os.path.getmtime(meta_path)
        if os.path.exists(meta_path) and t_meta > t_event:
            return np.load(meta_path)

        ea = event_accumulator.EventAccumulator(events_file, size_guidance={
            event_accumulator.SCALARS: 0, event_accumulator.IMAGES: 1,
            event_accumulator.TENSORS: 1})
        ea.Reload()

        out = {}
        for tag in ea.Tags()['scalars']:
            if not tag.startswith("debug"):
                scalar_events = ea.Scalars(tag)
                out[f"{tag}_i"] = np.array(
                    [x.step for x in scalar_events], dtype=np.int32)
                out[tag] = np.array([x.value for x in scalar_events])
        np.savez(meta_path, **out)
        return out
