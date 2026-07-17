"""Tests for `nrdk.config.config`."""

import logging
import os
import subprocess
import sys

import pytest
import yaml
from jaxtyping import TypeCheckError
from omegaconf import OmegaConf

from nrdk.config.config import (
    PreventHydraOverwrite,
    _SingletonRegistry,
    expand,
    inst_from,
)


def test_expand_single_level():
    """A single mapping-to-sequence level."""
    result = expand(a=["x.txt", "y.txt"])
    assert result == ["a/x.txt", "a/y.txt"]

    result = expand(path="base", a=["x.txt"])
    assert result == [os.path.join("base", "a", "x.txt")]


def test_expand_multiple_top_level_keys():
    """Multiple top-level keys are independently expanded and concatenated."""
    result = expand(path="root", a=["f1"], b=["f2"])
    assert sorted(result) == sorted([
        os.path.join("root", "a", "f1"),
        os.path.join("root", "b", "f2"),
    ])


def test_expand_nested_mapping_flattens_arbitrary_depth():
    """Deep nesting composes with a base `path` and sibling flat keys."""
    result = expand(path="root", a=["f1"], b={"c": {"d": ["f2"]}})
    assert sorted(result) == sorted([
        os.path.join("root", "a", "f1"),
        os.path.join("root", "b", "c", "d", "f2"),
    ])


def test_expand_malformed_input_raises_typecheck_error_in_normal_use():
    """Malformed input raises `TypeCheckError`, not `ValueError`.

    Through the normal, type-checked public API, malformed input never
    reaches `_expand`'s internal `ValueError` branches: it is rejected
    earlier as a `TypeCheckError` by beartype (see module-level note above).
    """
    with pytest.raises(TypeCheckError):
        expand(a=123)  # type: ignore
    with pytest.raises(TypeCheckError):
        expand(a=[1, 2, 3])  # type: ignore


def _run_without_typechecking(code: str) -> subprocess.CompletedProcess:
    """Run `code` in a fresh subprocess with jaxtyping/beartype disabled.

    Setting `JAXTYPING_DISABLE=1` before `nrdk.config` is first imported (in
    that subprocess) skips installing the runtime type-checking decorator,
    so `expand` behaves exactly like the plain Python defined in
    `config.py`.
    """
    env = dict(os.environ, JAXTYPING_DISABLE="1")
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, env=env, timeout=30)


def test_expand_malformed_input_raises_value_error_without_typechecking():
    """With type-checking disabled, malformed input reaches the real checks.

    Covers a non-`Sequence`/non-`Mapping` leaf, one nested under a further
    mapping, and a sequence leaf with non-`str` items.
    """
    code = (
        "from nrdk.config.config import expand\n"
        "for kwargs in [{'a': 123}, {'a': {'b': 123}}, {'a': [1, 2, 3]}]:\n"
        "    try:\n"
        "        expand(**kwargs)\n"
        "    except ValueError as e:\n"
        "        print(e)\n"
    )
    result = _run_without_typechecking(code)
    assert result.returncode == 0, result.stderr

    lines = result.stdout.splitlines()
    assert len(lines) == 3
    assert "Nested structure must be a sequence or mapping" in lines[0]
    assert "Nested structure must be a sequence or mapping" in lines[1]
    assert "All items in the last level" in lines[2]


def test_singleton_registry_register_populates_and_merges():
    """`register` populates the registry, and later calls update/overwrite."""
    registry = _SingletonRegistry()
    result = registry.register(a=1, b=2)
    assert result == {"a": 1, "b": 2}
    assert registry.singletons == {"a": 1, "b": 2}

    result = registry.register(a=3, c=4)
    assert result == {"a": 3, "c": 4}
    assert registry.singletons == {"a": 3, "b": 2, "c": 4}


def test_singleton_registry_call_retrieves_by_name():
    """Calling the registry with a name retrieves the registered object."""
    registry = _SingletonRegistry()
    sentinel = object()
    registry.register(thing=sentinel)

    assert registry("thing") is sentinel


def test_singleton_registry_missing_name_raises_key_error():
    """A missing name raises `KeyError` with a descriptive message."""
    registry = _SingletonRegistry()
    registry.register(known=1)

    with pytest.raises(KeyError) as exc_info:
        registry("unknown")

    message = str(exc_info.value)
    assert "No singleton registered with name unknown" in message
    assert "known" in message


def test_singleton_registry_fresh_instance_is_independent():
    """Two separate `_SingletonRegistry` instances do not share state."""
    r1 = _SingletonRegistry()
    r2 = _SingletonRegistry()
    r1.register(a=1)

    assert r1.singletons == {"a": 1}
    assert r2.singletons == {}


class _Widget:
    """Top-level, importable class used as a Hydra `_target_` in tests."""

    def __init__(self, value: int = 0) -> None:
        self.value = value


def _make_widget(value: int = 0) -> _Widget:
    """Top-level, importable function used as a Hydra `_target_` in tests."""
    return _Widget(value=value * 2)


def test_inst_from_whole_file(tmp_path):
    """`key=None` instantiates the entire configuration file."""
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(yaml.dump({
        "_target_": "tests.config.test_config._Widget",
        "value": 5,
    }))

    obj = inst_from(str(spec_path), None)

    assert isinstance(obj, _Widget)
    assert obj.value == 5


def test_inst_from_single_string_key(tmp_path):
    """A single string `key` selects the top-level entry to instantiate."""
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(yaml.dump({
        "widget": {
            "_target_": "tests.config.test_config._Widget",
            "value": 7,
        },
        "other": {"_target_": "tests.config.test_config._Widget"},
    }))

    obj = inst_from(str(spec_path), "widget")

    assert isinstance(obj, _Widget)
    assert obj.value == 7


def test_inst_from_sequence_of_keys(tmp_path):
    """A sequence of keys walks into a nested dict before instantiating."""
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(yaml.dump({
        "a": {
            "b": {
                "_target_": "tests.config.test_config._make_widget",
                "value": 3,
            },
        },
    }))

    obj = inst_from(str(spec_path), ["a", "b"])

    assert isinstance(obj, _Widget)
    assert obj.value == 6


def _hydra_config(run_dir):
    return OmegaConf.create({"hydra": {"run": {"dir": str(run_dir)}}})


def test_prevent_overwrite_missing_dir_is_noop(tmp_path):
    """A run dir that does not exist yet triggers no action."""
    run_dir = tmp_path / "does_not_exist"
    callback = PreventHydraOverwrite()

    callback.on_run_start(_hydra_config(run_dir))  # should not raise


def test_prevent_overwrite_dir_exists_without_checkpoints_is_noop(tmp_path):
    """An existing run dir with no `checkpoints` entry triggers no action."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "some_other_file.txt").write_text("hello")
    callback = PreventHydraOverwrite()

    callback.on_run_start(_hydra_config(run_dir))  # should not raise


def test_prevent_overwrite_dir_with_checkpoints_raises_system_exit(
    tmp_path, caplog,
):
    """An existing run dir with `checkpoints` logs an error and aborts."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "checkpoints").mkdir()
    callback = PreventHydraOverwrite()

    with caplog.at_level(logging.ERROR):
        with pytest.raises(SystemExit) as exc_info:
            callback.on_run_start(_hydra_config(run_dir))

    assert exc_info.value.code == 1
    assert "already exists and contains checkpoints" in caplog.text


def test_prevent_overwrite_path_is_file_raises_system_exit(tmp_path):
    """A run dir path that exists but is a file (not a dir) also aborts."""
    run_dir = tmp_path / "run"
    run_dir.write_text("not a directory")
    callback = PreventHydraOverwrite()

    with pytest.raises(SystemExit) as exc_info:
        callback.on_run_start(_hydra_config(run_dir))
    assert exc_info.value.code == 1


@pytest.mark.parametrize("var", ["RANK", "LOCAL_RANK", "GLOBAL_RANK"])
def test_prevent_overwrite_skipped_on_nonzero_rank(tmp_path, monkeypatch, var):
    """Non-rank-0 processes skip the check, even if checkpoints exist."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "checkpoints").mkdir()
    monkeypatch.setenv(var, "1")
    callback = PreventHydraOverwrite()

    callback.on_run_start(_hydra_config(run_dir))  # should not raise
