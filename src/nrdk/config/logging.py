"""Logging helpers."""

import logging

from rich.logging import RichHandler


def configure_rich_logging(level: int = logging.INFO) -> int:
    """Configure rich logging for the root logger.

    Analogous to `basicConfig(level=...)`.

    Args:
        level: log level to use.

    Returns:
        Configured log level; useful for chaining.
    """
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    rich_handler = RichHandler(markup=True)
    fmt = logging.Formatter("[orange1]%(name)s:[/orange1] %(message)s")
    rich_handler.setFormatter(fmt)
    root.addHandler(rich_handler)

    return level
