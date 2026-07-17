"""Tests for `nrdk.config.logging`."""

import logging

import pytest
from rich.logging import RichHandler

from nrdk.config.logging import _Rank0Filter, configure_rich_logging

# --------------------------------------------------------------------------
# _Rank0Filter
# --------------------------------------------------------------------------


def _make_record() -> logging.LogRecord:
    return logging.LogRecord(
        name="test", level=logging.INFO, pathname=__file__, lineno=1,
        msg="hello", args=None, exc_info=None)


def test_rank0_filter_passes_with_no_rank_vars_set(monkeypatch):
    """With no rank env vars set, the default of `0` for all three passes."""
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.delenv("GLOBAL_RANK", raising=False)

    assert _Rank0Filter().filter(_make_record()) is True


@pytest.mark.parametrize("var", ["RANK", "LOCAL_RANK", "GLOBAL_RANK"])
def test_rank0_filter_blocks_when_any_var_nonzero(monkeypatch, var):
    """Any single rank-like env var set to a nonzero value blocks it."""
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.delenv("GLOBAL_RANK", raising=False)
    monkeypatch.setenv(var, "1")

    assert _Rank0Filter().filter(_make_record()) is False


@pytest.mark.parametrize("var", ["RANK", "LOCAL_RANK", "GLOBAL_RANK"])
def test_rank0_filter_passes_when_var_explicitly_zero(monkeypatch, var):
    """Explicitly setting a rank var to `"0"` still passes the filter."""
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.delenv("GLOBAL_RANK", raising=False)
    monkeypatch.setenv(var, "0")

    assert _Rank0Filter().filter(_make_record()) is True


def test_rank0_filter_blocks_when_all_vars_nonzero(monkeypatch):
    """All three rank vars set to nonzero values blocks the record."""
    monkeypatch.setenv("RANK", "2")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("GLOBAL_RANK", "3")

    assert _Rank0Filter().filter(_make_record()) is False


# --------------------------------------------------------------------------
# configure_rich_logging()
# --------------------------------------------------------------------------


def test_configure_rich_logging_sets_level_and_returns_it():
    """The root logger's level is set, and the level is returned too."""
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    try:
        result = configure_rich_logging(level=logging.DEBUG)

        assert result == logging.DEBUG
        assert root.level == logging.DEBUG
    finally:
        root.handlers = saved_handlers
        root.setLevel(saved_level)


def test_configure_rich_logging_removes_stream_handlers_adds_rich():
    """Plain `StreamHandler`s are removed and a `RichHandler` is added."""
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    try:
        root.handlers = []
        stream_handler = logging.StreamHandler()
        file_handler = logging.NullHandler()
        root.addHandler(stream_handler)
        root.addHandler(file_handler)

        configure_rich_logging(level=logging.WARNING)

        assert stream_handler not in root.handlers
        assert file_handler in root.handlers
        assert any(
            isinstance(h, RichHandler) for h in root.handlers)
    finally:
        root.handlers = saved_handlers
        root.setLevel(saved_level)


def test_configure_rich_logging_default_level_is_info():
    """The default `level` argument is `logging.INFO`."""
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    try:
        result = configure_rich_logging()

        assert result == logging.INFO
        assert root.level == logging.INFO
    finally:
        root.handlers = saved_handlers
        root.setLevel(saved_level)


def test_configure_rich_logging_no_rank0_only_adds_no_filters():
    """An empty `rank0_only` list attaches no filters to any logger."""
    root = logging.getLogger()
    named_logger = logging.getLogger("some.other.logger")
    saved_handlers = list(root.handlers)
    saved_level = root.level
    saved_filters = list(named_logger.filters)
    try:
        configure_rich_logging()

        assert not any(
            isinstance(f, _Rank0Filter) for f in named_logger.filters)
    finally:
        root.handlers = saved_handlers
        root.setLevel(saved_level)
        named_logger.filters = saved_filters


def test_configure_rich_logging_rank0_only_attaches_filter():
    """Loggers named in `rank0_only` get a `_Rank0Filter` attached."""
    root = logging.getLogger()
    logger_a = logging.getLogger("pkg.module_a")
    logger_b = logging.getLogger("pkg.module_b")
    saved_handlers = list(root.handlers)
    saved_level = root.level
    saved_filters_a = list(logger_a.filters)
    saved_filters_b = list(logger_b.filters)
    try:
        configure_rich_logging(rank0_only=["pkg.module_a", "pkg.module_b"])

        assert any(
            isinstance(f, _Rank0Filter) for f in logger_a.filters)
        assert any(
            isinstance(f, _Rank0Filter) for f in logger_b.filters)
    finally:
        root.handlers = saved_handlers
        root.setLevel(saved_level)
        logger_a.filters = saved_filters_a
        logger_b.filters = saved_filters_b
