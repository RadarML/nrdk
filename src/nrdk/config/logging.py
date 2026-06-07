"""Logging helpers."""

import logging
import os

from rich.logging import RichHandler


class _Rank0Filter(logging.Filter):

    def filter(self, record: logging.LogRecord) -> bool:
        return (
            os.environ.get('RANK', '0') == '0' and
            os.environ.get('LOCAL_RANK', '0') == '0' and
            os.environ.get('GLOBAL_RANK', '0') == '0'
        )


def configure_rich_logging(
    level: int = logging.INFO, rank0_only: list[str] = []
) -> int:
    """Configure rich logging for the root logger.

    Analogous to `basicConfig(level=...)`.

    Args:
        level: log level to use.
        rank0_only: list of logger names that should only emit from (local)
            rank 0. If empty, no rank filtering is applied.

    Returns:
        Configured log level; useful for chaining.
    """
    root = logging.getLogger()
    root.setLevel(level)

    # Remove existing stdout handler to replace it with rich; other handlers
    # (e.g., file handlers from Hydra) are preserved.
    for handler in root.handlers[:]:
        if type(handler) is logging.StreamHandler:
            root.removeHandler(handler)

    rich_handler = RichHandler(markup=True)
    fmt = logging.Formatter("[orange1]%(name)s:[/orange1] %(message)s")
    rich_handler.setFormatter(fmt)
    root.addHandler(rich_handler)

    if rank0_only:
        for logger_name in rank0_only:
            logging.getLogger(logger_name).addFilter(_Rank0Filter())

    return level
