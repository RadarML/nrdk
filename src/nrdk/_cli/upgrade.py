"""Upgrade one implementation to another in hydra configs."""

import os
import re

import numpy as np
from rich import print
from rich.columns import Columns
from rich.panel import Panel

from nrdk.framework import Result


def _format_context(
    context: str, line_num: int | np.integer,
    start: int | np.integer, end: int | np.integer
) -> str:
    return '\n'.join([
        f"{'>>>' if n == line_num else '   '} {line}"
        for n, line in zip(range(start, end), context.split('\n'))
    ])


def _search(
    text: str, pattern: str | re.Pattern, context_size: int = 2
) -> list[tuple[int, str]]:
    if isinstance(pattern, str):
        pattern = re.compile(pattern)

    newlines = np.where(np.frombuffer(
        text.encode('utf-8'), dtype=np.uint8) == ord('\n'))[0]
    newlines = np.concatenate([[-1], newlines, [len(text)]])

    matches = []
    search_start = 0
    while True:
        match = pattern.search(text, search_start)
        if not match:
            break

        line_num = np.searchsorted(newlines, match.start(), side='right') - 1

        start = max(0, line_num - context_size)
        end = min(len(newlines) - 1, line_num + 1 + context_size)

        context = text[newlines[start] + 1:newlines[end]]
        matches.append(
            (line_num, _format_context(context, line_num, start, end)))

        search_start = match.end()

    return matches


def cli_upgrade(
    target: str, /, to: str | None = None,
    dry_run: bool = False, path: str = ".", follow_symlinks: bool = False
) -> None:
    """Upgrade implementation references in hydra configs.

    !!! info "Usage"

        First test with a dry run:
        ```sh
        nrdk upgrade-config <target> --path ./results --dry-run
        ```
        If you're happy with what you see, you can then run the actual upgrade:
        ```sh
        nrdk upgrade-config <target> <to> --path ./results
        ```

    !!! danger

        This is a potentially destructive operation! Always run with
        `--dry-run` first, and make sure that `to` does not overlap with
        any other existing implementations in your configs.

        You can also use the `upgrade-config` tool to check for this overlap
        first:
        ```sh
        nrdk upgrade-config <to> --path ./results --dry-run
        # Shouldn't return any of the config files you are planning to upgrade
        ```

    For each valid [results directory][nrdk.framework.Result] in the specified
    `path`, search for all `_target_` fields in the hydra config, and replace
    any occurrences of `from` with `to`.

    Args:
        target: full path name of the implementation to replace.
        to: full path name of the implementation to replace with.
        dry_run: if `True`, only log the changes that would be made, and do not
            actually modify any files.
        path: path to search for results directories.
        follow_symlinks: whether to follow symlinks when searching for results.
    """
    pattern = re.compile(rf"_target_\s*:\s*{re.escape(target)}(?=\s|$)")
    results = Result.find(path, follow_symlinks=follow_symlinks)

    if dry_run:
        all_matches = {}
        for r in results:
            config_path = os.path.join(r, ".hydra", "config.yaml")
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    config = f.read()

                matches = _search(config, pattern, context_size=2)
                for line_num, context in matches:
                    if context not in all_matches:
                        all_matches[context] = []
                    all_matches[context].append((config_path, line_num))

        for k, v in all_matches.items():
            print(
                f"Found {len(v)} occurrence(s) of '{target}' "
                f"with this context:")
            print(Panel(k))
            print(Columns(
                f"{os.path.relpath(config_path, path)}:{line_num}"
                for config_path, line_num in v))
            print()

    else:
        if to is None:
            raise ValueError("Must specify `to` when not doing a dry run.")

        for r in results:
            config_path = os.path.join(r, ".hydra", "config.yaml")
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    config = f.read()

                n = re.findall(pattern, config)
                if n:
                    print(f"Upgrading {len(n)} occurrence(s): {config_path}")
                    new_config = re.sub(pattern, f"_target_: {to}", config)
                    with open(config_path, "w") as f:
                        f.write(new_config)
