"""High level API for loading and calculating statistics from experiments."""

import os
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from multiprocessing import pool
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
import optree
import pandas as pd
from jaxtyping import Float64, Num
from scipy.stats import norm

from .stats import NDStats
from .utils import cut_trace, intersect_difference

LeafType = TypeVar("LeafType", bound=np.ndarray)

if TYPE_CHECKING:
    # NOTE: mkdocstrings uses TYPE_CHECKING mode, so we put the docstring here.
    NestedValues = Sequence["NestedValues"] | LeafType
    """An arbitrarily nested sequence, parameterized by a leaf type.

    For example, these are valid examples of
    `NestedValues[Float[np.ndarray, "_N"]]`:
    ```python
    nested_leaf = Float[np.ndarray, "N1"]
    nested_list = [Float[np.ndarray, "N1"], Float[np.ndarray, "N2"]]
    nested_list_list = [
        [Float[np.ndarray, "N1"], Float[np.ndarray, "N2"]],
        [Float[np.ndarray, "N3"], Float[np.ndarray, "N4"]],
    ]
    ```
    """
else:
    NestedValues = Sequence[Any] | LeafType


def index(
    path: str, pattern: str | re.Pattern, follow_symlinks: bool = False
) -> dict[str | None, dict[str | None, str]]:
    r"""Recursively find all evaluations matching the given pattern.

    !!! tip

        LLM chat bots are very good at writing simple regex patterns!

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


def _relative_stats(
    y: Mapping[str, NestedValues[Num[np.ndarray, "_N"]]],
    t: Mapping[str, NestedValues[Float64[np.ndarray, "_N"]]] | None,
    names: Sequence[str], baselines: Mapping[str, str],
    workers: int = -1, t_max: int | None = None
) -> NDStats:
    """Compute paired statistics against a per-experiment baseline.

    Args:
        y: mapping of experiment names and metric values.
        t: mapping of experiment names and timestamps; if provided, each
            difference is taken at timestamps common to the pair.
        names: experiments to compute statistics for, in output order.
        baselines: maps each entry of `names` to the experiment it should be
            compared against.
        workers: number of worker threads to use for computation.
        t_max: maximum time delay to consider when computing effective sample
            size; if `None`, do not use any additional constraints.

    Returns:
        Relative statistics, stacked along the leading axis in `names` order.
    """
    y_sorted = [y[k] for k in names]
    y_base = [y[baselines[k]] for k in names]
    if t is not None:
        t_sorted = [t[k] for k in names]
        t_base = [t[baselines[k]] for k in names]
        diff = optree.tree_map(
            intersect_difference,
            y_sorted, y_base, t_sorted, t_base)  # type: ignore
    else:
        diff = optree.tree_map(
            lambda a, b: a - b, y_sorted, y_base)  # type: ignore
    return NDStats.from_values(diff, workers=workers, t_max=t_max)  # type: ignore


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


@dataclass
class Control:
    r"""A control variable for paired comparisons against varying baselines.

    Passed to [`stats_from_controls`][^.] (or directly to
    [`dataframe_from_index`][^.]) to add a `rel_{name}/*` column group which
    is computed against a different baseline for each experiment, instead of
    the single global `baseline`.

    !!! info "Which factor `name` refers to"

        A control is named after the factor which is **held fixed**: each
        experiment is paired against a baseline which shares its value on
        that factor, so that factor cancels out of the difference, and
        `rel_{name}/*` reports the effect of everything *else*. In a
        `{ratio} x {split}` sweep, `rel_split/*` is the effect of the ratio,
        measured at matched splits.

        `name` is only a column label, so it is not checked: when several
        controls hold the *same* factor fixed but pair against different
        references, give them distinct labels (e.g. `split` and
        `split_vs_best`).

    !!! info

        Since each experiment is paired against its own baseline, no grouping
        is needed: `baselines` is simply a mapping from each experiment
        to the experiment it should be compared against.

    !!! warning "Percentages use the global baseline"

        `pct_{name}/*` is normalized by the *global* baseline's `abs/mean`,
        not by each group's own baseline, so that percentages stay comparable
        across every row of the dataframe.

    !!! example

        Comparing each experiment against the `base` variant of its family:
        ```python
        Control("family", {
            e: f"{e.rsplit('/', 1)[0]}/base" for e in index if e is not None})
        ```
        For the common case where the experiment name encodes a factor of a
        full-factorial sweep, use [`from_factor`][.] instead.

    Attributes:
        name: the factor held fixed by this control; names the resulting
            `rel_{name}/*`, `pct_{name}/*`, and `p0.05_{name}` columns.
        baselines: maps each experiment to the baseline it should be compared
            against for this control. Keys must be experiment names; an index
            built without an `experiment` group has a `None` key, which must
            be filtered out first. Experiments which are absent are excluded
            from this control (their `rel_{name}/*` columns are `NaN`); every
            baseline named must itself be present in the index.
    """

    name: str
    baselines: Mapping[str, str]

    @classmethod
    def from_factor(
        cls, name: str, pattern: str | re.Pattern,
        experiments: Iterable[str | None], reference: str
    ) -> "Control":
        r"""Create a control from one factor of a full-factorial sweep.

        Each experiment matching `pattern` is paired against `reference`, with
        `reference`'s own factor value replaced by the experiment's. This
        holds the remaining factors fixed at their reference values, so the
        resulting comparison controls for the factor captured by `pattern`.

        !!! tip

            `reference` does not have to be the global `baseline`: it is only
            the experiment which the factor value is substituted into, so it
            can be any experiment which matches `pattern` (e.g. the
            best-performing configuration, rather than the default one).

        !!! example

            For a sweep over `midtrain/t1_2k_p{ratio}_b4x8/{split}` with
            `reference="midtrain/t1_2k_p0.8_b4x8/p100"`:
            ```python
            # each experiment vs. the p0.8 model at the same split
            Control.from_factor(
                "split", r"^midtrain/t1_2k_p[\d.]+_b4x8/(?P<value>[^/]+)$",
                index, reference)
            # each experiment vs. the p100 split at the same ratio
            Control.from_factor(
                "ratio", r"^midtrain/t1_2k_p(?P<value>[\d.]+)_b4x8/p\d+$",
                index, reference)
            ```

        Args:
            name: name of the control; by convention, the factor captured
                by the `value` group, since that is the factor held fixed.
            pattern: regex matched against experiment names, defining a single
                `value` group which captures the controlled factor.
                Experiments which do not match are excluded, so `pattern` can
                also be used to restrict the control to a family of
                experiments.
            experiments: experiment names to build the mapping over; an
                [`index`][^^^.api.] can be passed directly.
            reference: the experiment which the factor value is substituted
                into. Must match `pattern`.

        Returns:
            A `Control` pairing each matching experiment with its baseline.
        """
        if isinstance(pattern, str):
            pattern = re.compile(pattern)
        if "value" not in pattern.groupindex:
            raise ValueError(
                f"Control '{name}': the factor pattern must define a `value` "
                f"group capturing the controlled factor; got: "
                f"{pattern.pattern}")

        matched = pattern.match(reference)
        if matched is None or matched.group("value") is None:
            raise ValueError(
                f"Control '{name}': the factor pattern does not match the "
                f"reference '{reference}' (or its `value` group does not "
                f"participate in the match), so the factor value cannot be "
                f"substituted; got: {pattern.pattern}")
        start, end = matched.span("value")

        names = [x for x in experiments if x is not None]
        baselines = {}
        for experiment in names:
            matched = pattern.match(experiment)
            if matched is not None:
                value = matched.group("value")
                if value is None:
                    raise ValueError(
                        f"Control '{name}': the `value` group did not "
                        f"participate in the match for experiment "
                        f"'{experiment}', so its factor value is undefined; "
                        f"got: {pattern.pattern}")
                baselines[experiment] = (
                    reference[:start] + value + reference[end:])

        if len(baselines) == 0:
            raise ValueError(
                f"Control '{name}': the factor pattern did not match any "
                f"experiments; got: {pattern.pattern}")

        if all(k == v for k, v in baselines.items()):
            raise ValueError(
                f"Control '{name}': every experiment is its own baseline, so "
                f"this control is a no-op. The `value` group should capture "
                f"the factor which is held fixed -- the one which the "
                f"experiment shares with its baseline -- and not the factor "
                f"which is compared; got: {pattern.pattern}")

        unknown = sorted(set(baselines.values()) - set(names))
        if len(unknown) > 0:
            raise ValueError(
                f"Control '{name}': substituting the factor value produced "
                f"baselines which are not in `experiments`: {unknown}. Check "
                f"that the sweep is full-factorial over this factor, and that "
                f"`pattern` captures the entire factor.")

        return cls(name=name, baselines=baselines)


@dataclass
class ControlStats:
    """Paired statistics computed for a single [`Control`][^.].

    Attributes:
        control: the control which these statistics were computed for.
        names: experiments covered by the control, corresponding to the
            leading axis of `stats`.
        stats: relative statistics, each experiment against its own baseline.
    """

    control: Control
    names: list[str]
    stats: NDStats


def stats_from_controls(
    y: Mapping[str, NestedValues[Num[np.ndarray, "_N"]]],
    t: Mapping[str, NestedValues[Float64[np.ndarray, "_N"]]] | None = None,
    controls: Sequence[Control] = (), workers: int = -1,
    t_max: int | None = None
) -> list[ControlStats]:
    """Calculate controlled statistics from experiment results.

    Computes one set of paired statistics per [`Control`][^.], each against
    that control's per-experiment baselines instead of a single global
    baseline; pass the result to [`dataframe_from_stats`][^.] to add the
    corresponding columns.

    Args:
        y: mapping of experiment names and metric values.
        t: mapping of experiment names and timestamps. If not provided, the
            metrics are assumed to be at identical timestamps.
        controls: control variables to compute paired statistics for.
        workers: number of worker threads to use for computation.
        t_max: maximum time delay to consider when computing effective sample
            size; if `None`, do not use any additional constraints.

    Returns:
        Statistics for each control, in the order provided.
    """
    _check_control_names([control.name for control in controls])

    names = sorted(y.keys())
    stats = []
    for control in controls:
        covered = [k for k in names if k in control.baselines]
        if len(covered) == 0:
            raise ValueError(
                f"Control '{control.name}' does not cover any of the loaded "
                f"experiments: {names}")

        missing = sorted({control.baselines[k] for k in covered} - set(names))
        if len(missing) > 0:
            raise ValueError(
                f"Control '{control.name}' refers to baselines which were not "
                f"loaded: {missing}. Check that these experiments are present "
                f"in the index, and are not excluded by `experiments`.")

        stats.append(ControlStats(
            control=control, names=covered, stats=_relative_stats(
                y, t, covered, control.baselines,
                workers=workers, t_max=t_max)))

    return stats


def _check_control_names(names: Sequence[str]) -> None:
    """Check that control names are unique.

    Args:
        names: names of the controls to check.
    """
    duplicates = sorted({k for k, v in Counter(names).items() if v > 1})
    if len(duplicates) > 0:
        raise ValueError(
            f"Controls must have unique names, since each control adds its "
            f"own column group; got duplicates: {duplicates}")


def _significant(
    mean: pd.Series, stderr: pd.Series,
    n_compared: "pd.Series | int"
) -> pd.Series:
    """Test whether each difference is significant at the 5% level.

    The test is two-sided, and is Bonferroni-corrected by the number of
    experiments compared against the same baseline.

    Args:
        mean: mean difference for each experiment.
        stderr: standard error of each difference.
        n_compared: number of non-baseline experiments sharing each baseline.

    Returns:
        Nullable boolean series, which is `pd.NA` for experiments where no
            comparison was made.
    """
    z = norm.ppf(1 - 0.05 / 2 / np.maximum(n_compared, 1))
    return (
        (mean.abs() / stderr) > z
    ).where(mean.notna()).astype("boolean")


def _append_control_columns(
    df: pd.DataFrame, control: ControlStats, baseline: str
) -> pd.DataFrame:
    """Add a single control's columns to a statistics dataframe.

    Args:
        df: statistics dataframe to add columns to, indexed by experiment.
        control: statistics for the control to add; see
            [`stats_from_controls`][^.].
        baseline: the global baseline, used as the denominator for the
            `pct_{name}/*` columns.

    Returns:
        The dataframe, with this control's columns added.
    """
    name = control.control.name
    prefix = f"rel_{name}/"
    df_rel = control.stats.reshape(
        len(control.names), -1).sum(axis=-1).as_df(control.names, prefix=prefix)
    df = df.merge(df_rel, on="name", how="left")

    _baseline = float(df.at[baseline, "abs/mean"])  # type: ignore
    df[f"pct_{name}/mean"] = df[f"{prefix}mean"] / _baseline * 100
    df[f"pct_{name}/stderr"] = df[f"{prefix}stderr"] / _baseline * 100

    # Bonferroni correction over the experiments sharing each baseline,
    # mirroring `dataframe_from_stats`' correction over all experiments.
    sizes = Counter(control.control.baselines[k] for k in control.names)
    n_compared = pd.Series({
        k: sizes[control.control.baselines[k]] - 1 for k in control.names
    }).reindex(df.index)
    df[f"p0.05_{name}"] = _significant(
        df[f"{prefix}mean"], df[f"{prefix}stderr"], n_compared)

    return df


def dataframe_from_stats(
    names: list[str], abs: NDStats, rel: NDStats | None = None,
    baseline: str | None = None, controls: Sequence[ControlStats] = ()
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

    Each control adds a `rel_{name}/*`, `pct_{name}/*`, and `p0.05_{name}`
    group with the same meaning, computed against that control's
    per-experiment baselines; see [`Control`][^.].

    Args:
        names: names of the experiments corresponding to the leading axis in
            the input statistics.
        abs: absolute statistics for the provided metric for each experiment.
        rel: optional relative statistics.
        baseline: name of the experiment used as the baseline.
        controls: optional controlled statistics; see
            [`stats_from_controls`][^.]. Requires a `baseline`.

    Returns:
        Dataframe with statistics for each experiment.
    """
    df = abs.reshape(
        len(names), -1).sum(axis=-1).as_df(names, prefix="abs/")

    if rel is not None and baseline is None:
        raise ValueError(
            "Provided relative statistics `rel`, but the `baseline` used is "
            "not specified.")

    if len(controls) > 0 and baseline is None:
        raise ValueError(
            "Provided `controls`, but no global `baseline`; a baseline is "
            "required to compute the `pct_{name}/*` columns.")

    _check_control_names([control.control.name for control in controls])
    for control in controls:
        extra = sorted(set(control.names) - set(names))
        if len(extra) > 0:
            raise ValueError(
                f"Control '{control.control.name}' has statistics for "
                f"experiments which are not in `names`: {extra}.")

    if rel is not None:
        df_rel = rel.reshape(
            len(names), -1).sum(axis=-1).as_df(names, prefix="rel/")
        df = df.merge(df_rel, on='name')
        _baseline = float(df.at[baseline, 'abs/mean'])  # type: ignore
        df['pct/mean'] = df['rel/mean'] / _baseline * 100
        df['pct/stderr'] = df['rel/stderr'] / _baseline * 100
        df['p0.05'] = _significant(
            df['rel/mean'], df['rel/stderr'], len(names) - 1)

    for control in controls:
        df = _append_control_columns(df, control, baseline)  # type: ignore

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
    (3) [`stats_from_controls`][^.], and (4) [`experiments_from_index`][^.].

    !!! tip "Controlling for a second variable"

        By default, every experiment is compared against a single global
        `baseline`, so the difference mixes together every axis along which
        the two differ. A [`Control`][^.] adds a second comparison which
        holds one of those axes fixed, pairing each experiment against the
        baseline sharing its value on that axis. Since the paired difference
        cancels that axis out, it generally yields a tighter standard error.

        Which axis to fix is a matter of the question being asked, not of
        which axis matters: pass one `Control` per axis to compare along
        each in turn.

        Each control adds a `rel_{name}/*`, `pct_{name}/*`, and
        `p0.05_{name}` column group, computed independently of the others.
        Experiments not covered by a control are `NaN` in its columns.

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

    y, t, _ = experiments_from_index(
        index, key, timestamps=timestamps, experiments=experiments,
        cut=cut, workers=workers)
    names, stats_abs, stats_rel = stats_from_experiments(
        y, t, baseline=baseline, workers=workers, t_max=t_max)
    stats_controls = stats_from_controls(
        y, t, controls=controls, workers=workers, t_max=t_max)
    return dataframe_from_stats(
        names, stats_abs, stats_rel, baseline=baseline,
        controls=stats_controls)
