"""Control variables for paired comparisons.

See [`dataframe_from_index`][^.api.] for what a control computes.
"""

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass


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
