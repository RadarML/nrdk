"""Create summary tables."""

import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import yaml

import analysis


def _parse(p):
    p.add_argument(
        "-p", "--path", default="results", help="Results directory.")
    p.add_argument(
        "--schema_dir", default="schema", help="Schema base directory.")
    p.add_argument(
        "-s", "--schema", default="base.yaml",
        help="Summary schema file.")
    p.add_argument(
        "-o", "--out", default="reports", help="Output directory.")


def _main(args):
    results = analysis.Results(args.path)

    if not args.schema.endswith(".yaml"):
        args.schema = args.schema + ".yaml"
    with open(os.path.join(args.schema_dir, args.schema)) as f:
        schema = yaml.load(f, Loader=yaml.FullLoader)

    for ablation, cfg in schema["comparisons"].items():
        rows = []
        methods = list(cfg["methods"].keys())

        for split, regex in schema["splits"].items():
            for metric in schema["metrics"]:
                compared = results.compare_to(
                    cfg["baseline"], methods, pattern=regex, key=metric,
                    allow_truncation=True).sum()

                z = zip(methods, compared.abs.mean, compared.diff.stderr)
                for method, mean, stderr in z:
                    rows.append({
                        "name": method, "split": split, metric: mean,
                        f"{metric}.stderr": np.nan_to_num(stderr),
                        **cfg["methods"][method]})

        df = pd.DataFrame(rows).groupby(
            ["name", "split"], as_index=False).first()
        df.to_csv(os.path.join("reports", ablation + ".csv"), index=False)


if __name__ == "__main__":
    p = ArgumentParser()
    _parse(p)
    _main(p.parse_args())
