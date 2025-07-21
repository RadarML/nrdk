"""Various CLI utilities for NRDK."""

from typing import Annotated, Any, Union

import tyro

from .export import cli_export
from .inspect import cli_inspect


def make_annotation(name, func):
    return Annotated[
        Any,
        tyro.conf.subcommand(
            name, description=func.__doc__.split('\n\n')[0],
            constructor=func
        )
    ]


def cli_main() -> None:
    commands = {
        "inspect": cli_inspect,
        "export": cli_export,
    }

    return tyro.cli(Union[  # type: ignore
        tuple(make_annotation(k, commands[k]) for k in sorted(commands.keys()))
    ])
