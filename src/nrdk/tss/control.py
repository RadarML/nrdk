"""Control variables for paired comparisons.

See [`dataframe_from_index`][^.api.] for what a control computes.
"""

from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import optree
import pandas as pd
from jaxtyping import Float64, Num
from scipy.stats import norm

from .stats import NDStats
from .utils import NestedValues, intersect_difference


@dataclass
class Control:
    r"""Specifications for a control variable.

    !!! warning

        Each control variable should have a unique name.

    Attributes:
        name: the name of this control variable.
        baselines: a mapping between experiment names and their respective
            baselines.
    """

    name: str
    baselines: Mapping[str, str]

    @classmethod
    def from_index_rule(
        cls, name: str, experiments: Iterable[str | None],
        rule: Callable[[str], str]
    ) -> "Control":
        """Create a control by applying a rule to each experiment name.

        !!! example

            Pairing each `<run>/<split>` experiment against the baseline run
            at the same split:
            ```python
            Control.from_index_rule(
                "split", index, lambda e: f"base/{e.rsplit('/', 1)[1]}")
            ```

        Args:
            name: the name of this control variable.
            experiments: experiment names to build the mapping over; an
                [`index`][^^^.api.] can be passed directly, and any `None`
                key is skipped.
            rule: given an experiment name, returns the name of the
                experiment which it should be compared against.

        Returns:
            A control which pairs each experiment with `rule(experiment)`.
        """
        return cls(name, {e: rule(e) for e in experiments if e is not None})

    def resolve(self, experiments: Sequence[str]) -> list[str]:
        """Experiments which this control applies to.

        Args:
            experiments: all experiment names which were loaded.

        Returns:
            The experiments which this control pairs against a baseline, in
                the order given.

        Raises:
            ValueError: if this control covers none of `experiments`, or
                names a baseline which is not among them.
        """
        covered = [k for k in experiments if k in self.baselines]
        if len(covered) == 0:
            raise ValueError(
                f"Control '{self.name}' does not cover any of the available "
                f"experiments: {list(experiments)}")

        missing = sorted(
            {self.baselines[k] for k in covered} - set(experiments))
        if len(missing) > 0:
            raise ValueError(
                f"Control '{self.name}' refers to baselines which are not "
                f"present in the index: {missing}.")

        return covered


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


def _append_control(
    df: pd.DataFrame, control: Control,
    y: Mapping[str, NestedValues[Num[np.ndarray, "_N"]]],
    t: Mapping[str, NestedValues[Float64[np.ndarray, "_N"]]] | None,
    names: Sequence[str], baseline: str, workers: int = -1,
    t_max: int | None = None
) -> pd.DataFrame:
    """Add a single control's columns to a statistics dataframe.

    Args:
        df: statistics dataframe to add columns to, indexed by experiment.
        control: the control variable to compute; see [`Control`][^.].
        y: mapping of experiment names and metric values.
        t: mapping of experiment names and timestamps.
        names: all loaded experiment names.
        baseline: the global baseline, used as the denominator for the
            `pct_{name}/*` columns.
        workers: number of worker threads to use for computation.
        t_max: maximum time delay to consider when computing effective sample
            size.

    Returns:
        The dataframe, with this control's columns added.
    """
    covered = control.resolve(names)
    rel = _relative_stats(
        y, t, covered, control.baselines, workers=workers, t_max=t_max)
    prefix = f"rel_{control.name}/"
    df_rel = rel.reshape(
        len(covered), -1).sum(axis=-1).as_df(covered, prefix=prefix)
    df = df.merge(df_rel, on="name", how="left")

    _baseline = float(df.at[baseline, "abs/mean"])  # type: ignore
    df[f"pct_{control.name}/mean"] = df[f"{prefix}mean"] / _baseline * 100
    df[f"pct_{control.name}/stderr"] = df[f"{prefix}stderr"] / _baseline * 100

    # As in `dataframe_from_stats`, but Bonferroni-corrected by the number of
    # experiments sharing each baseline instead of over all of them.
    sizes = Counter(control.baselines[k] for k in covered)
    n_compared = pd.Series({
        k: sizes[control.baselines[k]] - 1 for k in covered
    }).reindex(df.index)
    z = norm.ppf(1 - 0.05 / 2 / np.maximum(n_compared, 1))
    df[f"p0.05_{control.name}"] = (
        (df[f"{prefix}mean"].abs() / df[f"{prefix}stderr"]) > z
    ).where(df[f"{prefix}mean"].notna()).astype("boolean")

    return df
