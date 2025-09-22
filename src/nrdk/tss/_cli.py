"""CLI interface for calculating statistics."""

from io import StringIO

import tyro
import yaml

from . import api


def _cli(
    path: str, /,
    pattern: str | None = None, key: str | None = None,
    timestamps: str | None = None, experiments: list[str] | None = None,
    baseline: str | None = None, follow_symlinks: bool = False,
    cut: float | None = None, t_max: int | None = None,
    config: str | None = None,
) -> int:
    """Calculate statistics for time series metrics.

    - pipe `tss ... > results.csv` to save the results to a file.
    - use `--config config.yaml` to avoid having to specify all these
        arguments; any arguments which are explicitly provided will override
        the values in the config file.

    !!! warning

        `path` (and `--follow_symlinks`, if specified) are required to be
        passed via the command line, and cannot be specified via the config.

    Args:
        path: directory to find evaluations in.
        pattern: regex pattern to match the evaluation directories.
        key: name of the metric to load from the result files.
        timestamps: name of the timestamps to load from the result files.
        experiments: explicit list of experiments to include in the results.
        baseline: baseline experiment for relative statistics.
        follow_symlinks: whether to follow symbolic links. May lead to infinite
            recursion if `True` and the `path` contains self-referential links!
        cut: cut each time series when there is a gap in the timestamps larger
            than this value if provided.
        t_max: maximum time delay to consider when computing effective sample
            size; if not specified, do not use any additional constraints.
        config: load all of these values from a yaml configuration file
            instead.
    """
    if config is not None:
        with open(config) as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {}

    def setdefault(value, param, default):
        if value is None:
            value = cfg.get(param, default)
        return value

    pattern = setdefault(pattern, "pattern", r"^(?P<experiment>(.*)).npz$")
    key = setdefault(key, "key", "loss")
    timestamps = setdefault(timestamps, "timestamps", None)
    baseline = setdefault(baseline, "baseline", None)
    cut = setdefault(cut, "cut", None)
    t_max = setdefault(t_max, "t_max", None)

    index = api.index(
        path, pattern=pattern, follow_symlinks=follow_symlinks)  # type: ignore

    if len(index) == 0:
        print("No result files found!")
        print(
            "Hint: if `results` include symlinks (or is a symlink itself), "
            "try passing `--follow_symlinks`.")
        return -1

    df = api.dataframe_from_index(
        index, key=key, baseline=baseline,  # type: ignore
        experiments=experiments, cut=cut, t_max=t_max, timestamps=timestamps)

    buf = StringIO()
    df.to_csv(buf)
    print(buf.getvalue())
    return 0


def _cli_main() -> int:
    return tyro.cli(_cli)
