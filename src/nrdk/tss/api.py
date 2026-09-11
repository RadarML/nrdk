"""High level API for loading and calculating statistics from experiments."""

import os
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from multiprocessing import pool

import numpy as np
import pandas as pd
from jaxtyping import Float64, Num
from scipy.stats import norm

from .control import Control, _append_control, _relative_stats
from .stats import NDStats
from .utils import NestedValues, cut_trace


def index(
    path: str, pattern: str | re.Pattern, follow_symlinks: bool = False
) -> dict[str | None, dict[str | None, str]]:
    r"""Recursively find all evaluations matching the given pattern.

    The pattern can have two groups: `experiment`, and `trace`, which
    respectively indicate the name of the experiment and trace. If either group
    is omitted, it is set as `None`.

    !!! example

        ```python
        # match `<experiment>/eval/<trace>.npz`
        index(path, r'^(?P<experiment>(.*))/eval/(?P<trace>(.*))\.npz$')
        # match `<experiment>.npz`
        index(path, r'^(?P<experiment>(.*))\.npz$')
        ```

    Args:
        path: directory to start searching from.
        pattern: regex pattern to match the evaluation directories.
        follow_symlinks: whether to follow symbolic links.

    Returns:
        A two-level dictionary, where the first level keys are the experiment
            names, the second level keys are the trace names, and the values
            are paths to the matching files.
    """
    if isinstance(pattern, str):
        pattern = re.compile(pattern)
    manifest = {}

    def _find(path, base):
        matches = pattern.match(os.path.relpath(path, base))
        if matches is not None:
            groups = matches.groupdict()
            manifest.setdefault(
                groups.get('experiment', None), {}
            )[groups.get('trace', None)] = path
        elif os.path.isdir(path):
            if follow_symlinks or not os.path.islink(path):
                for p in os.listdir(path):
                    _find(os.path.join(path, p), base)

    _find(path, path)
    return manifest


def experiments_from_index(
    index: dict[str | None, dict[str | None, str]],
    key: str, timestamps: str | None = None,
    experiments: Sequence[str | None] | str | None = None,
    cut: float | None = None, workers: int = -1
) -> tuple[
    Mapping[str, NestedValues[Num[np.ndarray, "_N"]]],
    Mapping[str, NestedValues[Float64[np.ndarray, "_N"]]] | None,
    list[str | None]
]:
    """Load experiment results from indexed result files.

    Each results file is expected to be a `.npz` file containing metric and
    metadata arrays; the keys for these arrays should be specified by `key` and
    `timestamps`, respectively.

    - These arrays should all have the same leading axis length.
    - The metric array should have only a single axis.

    !!! warning

        Only sequences which are present in all experiments will be loaded.
        Check the returned `common` list to make sure it matchse what you
        expect!

    !!! tip

        A `timestamps` key can optionally be provided.

        - If not provided, the metrics are assumed to be at identical
            timestamps.
        - If multiple timestamps are present, the last one is used.

    Args:
        index: 2-level dictionary with experiment names, sequence/trace names,
            and paths to the result files; see [`index`][^.].
        key: name of the metric to load from the result files.
        timestamps: name of the timestamps to load from the result files.
        experiments: list of experiment names to load from the index (or a
            regex filter); loads all experiments if not specified.
        cut: cut each time series when there is a gap in the timestamps larger
            than this value if provided; see [`cut_trace`][^^.utils.].
        workers: number of worker threads to use when loading. If `<0`, load
            all in parallel; if `=0`, load all in the main thread.

    Returns:
        A dictionary of metric values (as a list of metric values by sequence).
        A dictionary of timestamps (or `None` if not specified).
        A list of the common sequence/trace names which correspond to the
            loaded metrics.
    """
    if len(index) == 0:
        raise ValueError("Could not fetch experiments: the index is empty.")

    if experiments is None:
        experiments = list(index.keys())
    elif isinstance(experiments, str):
        re_filter = re.compile(experiments)
        experiments = [x for x in index.keys() if re_filter.match(str(x))]
        if len(experiments) == 0:
            raise ValueError(
                f"No experiments found matching the filter: {experiments}")

    common = list(set.intersection(
        *[set(index[k].keys()) for k in experiments]))
    if workers < 0:
        workers = len(common) * len(experiments)

    def _load(path: str):
        data = np.load(path)

        if key not in data:
            raise KeyError(
                f"Key '{key}' not found in file: {path}. Available keys: "
                f"{list(data.keys())}")

        if timestamps is not None:
            t = data[timestamps]
            t = t.reshape(t.shape[0], -1)[:, -1]
            if cut is not None:
                ytyt = cut_trace(t, (data[key], t), gap=cut)
                return list(zip(*ytyt))
            else:
                return [data[key]], [t]
        else:
            return [data[key]]

    iterload = [(x, s) for x in experiments for s in common]
    if workers == 0:
        loaded = [_load(index[x][s]) for x, s in iterload]
    else:
        with pool.ThreadPool(workers) as p:
            loaded = list(p.map(_load, [index[x][s] for x, s in iterload]))

    if timestamps is not None:
        yy, tt = {}, {}
        for (x, s), (y, t) in zip(iterload, loaded):
            yy.setdefault(x, []).extend(y)
            tt.setdefault(x, []).extend(t)
        return yy, tt, common
    else:
        yy = {}
        for (x, s), y in zip(iterload, loaded):
            yy.setdefault(x, []).extend(y)
        return yy, None, common


def stats_from_experiments(
    y: Mapping[str, NestedValues[Num[np.ndarray, "_N"]]],
    t: Mapping[str, NestedValues[Float64[np.ndarray, "_N"]]] | None = None,
    baseline: str | None = None, workers: int = -1, t_max: int | None = None
) -> tuple[list[str], NDStats, NDStats | None]:
    """Calculate statistics from experiment results.

    Args:
        y: mapping of experiment names and metric values.
        t: mapping of experiment names and timestamps. If not provided, the
            metrics are assumed to be at identical timestamps.
        baseline: baseline experiment for relative statistics.
        workers: number of worker threads to use for computation.
        t_max: maximum time delay to consider when computing effective sample
            size; if `None`, do not use any additional constraints.

    Returns:
        Names of each experiment corresponding to leading axis in the output
            statistics.
        Absolute statistics for the provided metric.
        Relative statistics (difference relative to the specified baseline), if
            provided.
    """
    n_sorted = sorted(y.keys())
    if t is not None and set(n_sorted) != set(t.keys()):
        raise ValueError(
            f"Keys of `y` and `t` must match if `t` is provided: "
            f"y:{list(y.keys())}, t:{list(t.keys())}")

    y_sorted = [y[k] for k in n_sorted]
    stats_abs = NDStats.from_values(y_sorted, workers=workers, t_max=t_max)
    if baseline is not None:
        stats_rel = _relative_stats(
            y, t, n_sorted, {k: baseline for k in n_sorted},
            workers=workers, t_max=t_max)
    else:
        stats_rel = None
    return n_sorted, stats_abs, stats_rel


def dataframe_from_stats(
    names: list[str], abs: NDStats, rel: NDStats | None = None,
    baseline: str | None = None
) -> pd.DataFrame:
    """Create a dataframe from (possibly un-aggregated) experiment statistics.

    Returns a dataframe where each row is a different experiment.

    - `abs/(mean|std|stderr|n|ess)`: absolute statistics for the
        provided metric for each experiment.
    - `rel/(mean|std|stderr|n|ess)`: relative statistics for the
        provided metric for each experiment, relative to the `baseline`. If no
        `baseline` is provided, these columns are not included.
    - `pct/(mean|stderr)`: percent difference and standard error relative to
        the `baseline`, computed as `100 * <rel/mean>/<abs/mean>` and
        `100 * <rel/stderr>/<abs/mean>`, where `<abs/mean>` is the
        *baseline's* absolute mean.
    - `p0.05`: whether the difference from the `baseline` is significant at
        the 5% level (two-sided), Bonferroni-corrected by the number of
        experiments compared against the baseline; `pd.NA` where no
        comparison was made.

    Args:
        names: names of the experiments corresponding to the leading axis in
            the input statistics.
        abs: absolute statistics for the provided metric for each experiment.
        rel: optional relative statistics.
        baseline: name of the experiment used as the baseline.

    Returns:
        Dataframe with statistics for each experiment.
    """
    df = abs.reshape(
        len(names), -1).sum(axis=-1).as_df(names, prefix="abs/")

    if rel is not None and baseline is None:
        raise ValueError(
            "Provided relative statistics `rel`, but the `baseline` used is "
            "not specified.")

    if rel is not None:
        df_rel = rel.reshape(
            len(names), -1).sum(axis=-1).as_df(names, prefix="rel/")
        df = df.merge(df_rel, on='name')
        _baseline = float(df.at[baseline, 'abs/mean'])  # type: ignore
        df['pct/mean'] = df['rel/mean'] / _baseline * 100
        df['pct/stderr'] = df['rel/stderr'] / _baseline * 100

        # Two-sided, Bonferroni-corrected by the number of experiments
        # compared against the baseline
        z = norm.ppf(1 - 0.05 / 2 / max(len(names) - 1, 1))
        df['p0.05'] = (
            (df['rel/mean'].abs() / df['rel/stderr']) > z
        ).where(df['rel/mean'].notna()).astype("boolean")

    return df


def dataframe_from_index(
    index: dict[str | None, dict[str | None, str]],
    key: str, timestamps: str | None = None,
    experiments: Sequence[str | None] | None = None,
    cut: float | None = None, baseline: str | None = None, workers: int = -1,
    t_max: int | None = None, controls: Sequence[Control] = ()
) -> pd.DataFrame:
    """Load and calculate statistics from indexed experiment results.

    See (1) [`dataframe_from_stats`][^.], (2) [`stats_from_experiments`][^.],
    and (3) [`experiments_from_index`][^.].

    !!! info "Controlling for additional variables"

        By default, every experiment is compared against a single global
        `baseline`. A [`Control`][^.] adds a second comparison which
        holds one of those axes fixed, pairing each experiment against the
        baseline sharing its value on that axis.

        Each control adds a `rel_{name}/*`, `pct_{name}/*`, and
        `p0.05_{name}` column group, computed independently of the others,
        which report the effect of every axis *except* the one it holds
        fixed. Experiments which a control does not cover are `NaN` in its
        columns, and every baseline it names must itself be loaded.

    Args:
        index: 2-level dictionary with experiment names, sequence/trace names,
            and paths to the result files; see [`index`][^^.api.].
        key: name of the metric to load from the result files.
        timestamps: name of the timestamps to load from the result files.
        experiments: list of experiment names to load from the index; loads all
            experiments if not specified.
        cut: cut each time series when there is a gap in the timestamps larger
            than this value if provided; see [`cut_trace`][^^.utils.].
        baseline: baseline experiment for relative statistics.
        workers: number of worker threads to use when loading. If `<0`, load
            all in parallel; if `=0`, load all in the main thread.
        t_max: maximum time delay to consider when computing effective sample
            size; if `None`, do not use any additional constraints.
        controls: additional control variables to compute paired statistics
            for; see [`Control`][^.]. Requires a `baseline`.

    Returns:
        Dataframe with statistics for each experiment.
    """
    if len(controls) > 0 and baseline is None:
        raise ValueError(
            "Provided `controls`, but no global `baseline`; a baseline is "
            "required to compute the `pct_{name}/*` columns.")

    duplicates = sorted({
        k for k, v in Counter(c.name for c in controls).items() if v > 1})
    if len(duplicates) > 0:
        raise ValueError(
            f"Controls must have unique names, since each control adds its "
            f"own column group; got duplicates: {duplicates}")

    y, t, _ = experiments_from_index(
        index, key, timestamps=timestamps, experiments=experiments,
        cut=cut, workers=workers)
    names, stats_abs, stats_rel = stats_from_experiments(
        y, t, baseline=baseline, workers=workers, t_max=t_max)
    df = dataframe_from_stats(names, stats_abs, stats_rel, baseline=baseline)

    for control in controls:
        df = _append_control(
            df, control, y, t, names, baseline,  # type: ignore
            workers=workers, t_max=t_max)

    return df
