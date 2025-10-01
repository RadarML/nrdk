"""Validate results directories."""

import os

from rich import print
from rich.table import Table

from nrdk.framework import Result


def cli_validate(
    path: str, /, follow_symlinks: bool = False, show_all: bool = False,
) -> None:
    """Validate results directories.

    !!! info "Usage"

        ```sh
        nrdk validate <path> --follow_symlinks
        ```

    For each valid [results directory][nrdk.framework.Result] in the specified
    `path`, check that all expected files are present:

    | File                    | Description                                   |
    | ----------------------- | --------------------------------------------- |
    | `.hydra/config.yaml`    | Hydra configuration used for the run.         |
    | `checkpoints/last.ckpt` | Last model checkpoint saved during training.  |
    | `eval/`                 | Directory containing evaluation outputs.      |
    | `checkpoints.yaml` | Checkpoint index; absence indicates a crashed run. |
    | `events.out.tfevents.*` | Tensorboard log files.                        |

    Args:
        path: path to search for results directories.
        follow_symlinks: whether to follow symlinks when searching for results.
        show_all: show all results instead of just results with missing files.
    """
    results = Result.find(path, follow_symlinks=follow_symlinks, strict=False)

    _check_files = [
        ".hydra/config.yaml",
        "checkpoints/last.ckpt",
        "eval",
        "checkpoints.yaml",
    ]
    _status = {
        True: u'[green]\u2713[/green]',
        False: u'[bold red]\u2718[/bold red]',
    }

    missing = 0

    table = Table()
    table.add_column("path", justify="right", style="cyan")
    table.add_column("config.yaml", justify="left")
    table.add_column("last.ckpt", justify="left")
    table.add_column("eval", justify="left")
    table.add_column("checkpoints.yaml", justify="left")
    table.add_column("tfevents", justify="left")

    for r in results:
        row = [
            os.path.exists(os.path.join(r, file))
            for file in _check_files
        ] + [any(
            fname.startswith("events.out.tfevents.")
            for fname in os.listdir(r)
        )]
        if not all(row):
            missing += 1
        if show_all or not all(row):
            table.add_row(os.path.relpath(r, path), *[_status[x] for x in row])

    if missing > 0:
        print(
            f"Found {len(results)} results directories with {missing} "
            f"incomplete results.")
    else:
        print(f"All {len(results)} results directories are complete.")

    print(table)
