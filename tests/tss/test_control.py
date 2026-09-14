"""Tests for `nrdk.tss.control`."""

import pytest

from nrdk.tss.control import Control


def test_from_index_rule_applies_the_rule_to_each_experiment():
    """Each experiment is paired with whatever the rule names."""
    experiments = ["fama/p10", "famb/p10", "famb/p100"]

    control = Control.from_index_rule(
        "split", experiments, lambda e: f"base/{e.rsplit('/', 1)[1]}")

    assert control.name == "split"
    assert control.baselines == {
        "fama/p10": "base/p10",
        "famb/p10": "base/p10",
        "famb/p100": "base/p100",
    }


def test_from_index_rule_accepts_an_index_directly():
    """`index`-shaped dicts (whose keys may be `None`) can be passed as-is."""
    index = {"fam/a": {}, "fam/b": {}, None: {}}

    control = Control.from_index_rule("fam", index, lambda e: "fam/a")

    assert control.baselines == {"fam/a": "fam/a", "fam/b": "fam/a"}


def test_control_resolve_follows_the_given_order():
    """Only experiments in the mapping are covered, in the order given."""
    control = Control("split", {
        "famb/p10": "fama/p10", "fama/p10": "fama/p10"})

    covered = control.resolve(["fama/p10", "famb/p10", "other/p10"])

    assert covered == ["fama/p10", "famb/p10"]


def test_control_resolve_raises_when_nothing_is_covered():
    """A control which pairs none of the loaded experiments is a mistake."""
    control = Control("split", {"nothere/p10": "fama/p10"})

    with pytest.raises(ValueError, match="does not cover any of the available"):
        control.resolve(["fama/p10", "famb/p10"])


def test_control_resolve_raises_on_an_absent_baseline():
    """Baselines must themselves be loaded, or there is nothing to subtract."""
    control = Control("split", {"famb/p10": "fama/nope"})

    with pytest.raises(ValueError, match="baselines which are not present"):
        control.resolve(["fama/p10", "famb/p10"])
