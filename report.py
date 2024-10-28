"""Generate Statistical Reports."""

import os
from argparse import ArgumentParser

import yaml
from beartype.typing import Optional
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

import analysis


def _parse(p):
    p.add_argument(
        "-m", "--methods", nargs='+', default=None,
        help="Method comparison sets to generate plots for. Generates all by "
        "default.")
    p.add_argument(
        "-p", "--path", default="results", help="Results directory.")
    p.add_argument(
        "--schema_dir", default="schema", help="Schema base directory.")
    p.add_argument(
        "-s", "--schema", default="base.yaml",
        help="Report schema file.")
    p.add_argument(
        "-o", "--out", default="reports", help="Output directory.")


def _compare(
    results, methods: list[str], method_names: list[str],
    splits: dict[str, str], metric: str = "bev_loss",
    title: Optional[str] = None,
    cmap: str = "coolwarm"
):
    compared = {
        k: results.compare(methods, key=metric, pattern=v)
        for k, v in splits.items()}

    size = 0.75 * len(methods)
    figsize = (1.5 + len(compared) * size, 1.5 + 1.5 * size)
    fig, axs = plt.subplots(3, len(compared), figsize=figsize)
    axs = axs.reshape(3, len(compared))

    analysis.comparison_grid(
        axs, compared, method_names, aspect='auto', cmap=cmap,
        shortnames=all(len(n) <= 10 for n in method_names))

    for k, row in zip(compared, axs.T):
        row[-1].set_xlabel(k)

    if title is not None:
        fig.suptitle(title)
    fig.tight_layout()

    return fig


def _main(args):
    results = analysis.Results(args.path)

    plt.switch_backend('pdf')

    if not args.schema.endswith(".yaml"):
        args.schema = args.schema + ".yaml"
    with open(os.path.join(args.schema_dir, args.schema)) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    if args.methods is None:
        args.methods = list(cfg["methods"].keys())

    desc = args.schema.replace('.yaml', '') + ":" + ",".join(args.methods)
    for set in tqdm(args.methods, desc=desc):
        spec = cfg["methods"][set]
        with PdfPages(os.path.join(args.out, set + ".pdf")) as document:
            for metric, metric_name in  cfg["metrics"].items():
                fig = _compare(
                    results,
                    methods=spec["methods"],
                    method_names=spec["descriptions"],
                    splits=cfg["splits"],
                    metric=metric,
                    title=f"{spec['name']} / {metric_name}",
                    cmap="coolwarm")
                document.savefig(fig)
                plt.close(fig)


if __name__ == "__main__":
    p = ArgumentParser()
    _parse(p)
    _main(p.parse_args())
