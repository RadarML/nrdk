"""Compute statistics."""

import numpy as np
import os
from matplotlib import pyplot as plt
from multiprocessing import Pool
from scipy.stats import norm

from jaxtyping import Num, Float


def effective_sample_size(x: Num[np.ndarray, "t"]) -> float:
    """Calculate effective sample size for time series data."""
    with np.errstate(divide='ignore', invalid='ignore'):
        rho = np.array([
            np.cov(x[i:], x[:-i])[0, 1] / np.std(x[i:]) / np.std(x[:-i])
            for i in range(1, x.shape[0] // 2)])
    rho_sum = np.sum(np.maximum(0.0, rho))
    return x.shape[0] / (1 + 2 * rho_sum)


def load_single(path: str) -> dict[str, dict[str, Num[np.ndarray, "t ..."]]]:
    """Load a single evaluation."""
    data = {}
    for root, _, files in os.walk(os.path.join(path, "eval")):
        for file in files:
            npz = dict(np.load(os.path.join(root, file)))
            data[file.replace('.npz', '')] = npz
    return data


def load_array(
    results: list[str]
) -> dict[str, dict[str, Num[np.ndarray, "nr t ..."]]]:
    """Load all evaluations across results as arrays."""
    traces = [load_single(x) for x in results]
    trace_names = sorted(traces[0].keys())
    data = {}
    for name in trace_names:
        if all(name in t for t in traces):
            try:
                data[name] = {
                    metric: np.stack([t[name][metric] for t in traces], axis=0)
                    for metric in traces[0][name]}
            except Exception as e:
                breakpoint()

    return data


def _calculate_stats(trace, key: str = "chamfer"):
    diff = trace[key][None, :, :] - trace[key][:, None, :]

    with Pool() as p:
        ess = np.array(
            p.map(effective_sample_size, diff.reshape(-1, diff.shape[-1]))
        ).reshape(diff.shape[:2])

    return {
        "ess": ess, "n": diff.shape[-1],
        "m1": np.sum(diff, axis=2), "m2": np.sum(diff**2, axis=2)}


def calculate_stats(
    data: dict[str, dict[str, Num[np.ndarray, "nr t ..."]]],
    key: str = "chamfer"
) -> tuple[
    dict[str, Float[np.ndarray, "nr nr"]],
    dict[str, Float[np.ndarray, "..."]]
]:
    """Compute statistics."""
    _stats = [_calculate_stats(v, key=key) for v in data.values()]
    stats = {k: np.stack([x[k] for x in _stats]) for k in _stats[0]}

    n = np.sum(stats["n"])
    ess = np.sum(stats["ess"], axis=0)
    mean = np.sum(stats["m1"], axis=0) / n
    std = np.sqrt(np.sum(stats["m2"], axis=0) / n - mean**2)

    with np.errstate(divide='ignore'):
        stderr = std / np.sqrt(ess)
        zscore = mean / stderr

    summary = {
        "mean": mean, "std": std, "ess": ess,
        "stderr": stderr, "zscore": zscore
    }

    return summary, stats


import sys
statname = sys.argv[1]


results = [
    "radarhd/baseline", "radarhd/doppler", "radarhd/unweighted",
    "radartransformer/baseline", "radartransformer/aug-ctd",
    "radartransformer/large", "radartransformer/large-aug",
    "radartransformer/xlarge-ctd"]

data = load_array([os.path.join("results", r) for r in results])

summary, stats = calculate_stats(data, key=statname)


def color_significance(
    x: Float[np.ndarray, "n n"], z1: float, z2: float
) -> Float[np.ndarray, "n n"]:
    """Color significance for a two-sided statistical test.

    Args:
        x: array of test statistic values.
        z1, z2: test statistic thresholds.
    Returns:
        Discretized significance values. Values which exceed both thresholds
        (on either side) return values of `+/-2`; values exceeding only one
        return `+/-1`. Values under both thresholds return `0`.
    """
    res = -1.0 * (x < -z2) - 1.0 * (x < -z1) + 1.0 * (x > z1) + 1.0 * (x > z2)
    res[np.isnan(x)] = np.nan
    return res


def _matshow(ax, x, cm):
    ax.imshow(x, cmap=cm)
    for (i, j), label in np.ndenumerate(x):
        if abs(label) > 100.0:
            ax.text(i, j, f"{label:.0f}", ha='center', va='center')
        elif abs(label) > 10.0:
            ax.text(i, j, f"{label:.1f}", ha='center', va='center')
        else:
            ax.text(i, j, f"{label:.2f}", ha='center', va='center')
    ax.set_xticks(np.arange(len(results)))
    ax.set_yticks(np.arange(len(results)))
    ax.set_yticklabels([])
    ax.set_xticklabels([])

_stats = {
    "Effective Sample Size": (summary["ess"], "viridis"),
    "Difference": (summary["mean"], "GnBu"),
    "Standard Error": (summary["stderr"], "viridis"),
    "Z-score*": (summary["zscore"], 'GnBu'),
}

# 2-sided test + bonferroni correction
p_target = 0.05 / 2
bonferroni = p_target / ((len(results) * (len(results) - 1)) / 2)
z_uc_cutoff = norm.ppf(1 - p_target)
z_cutoff = norm.ppf(1 - bonferroni)

labels = [
    x.replace("radarhd", "hd").replace("radartransformer", "rxf")
    for x in results]

size = 1.5 + 1.0 * len(results)
fig, axs = plt.subplots(2, 2, figsize=(size, size))
for ax, (sn, (arr, cm)) in zip(axs.reshape(-1), _stats.items()):
    _matshow(ax, arr, cm)
    ax.set_title(sn)

axs[1, 1].imshow(-color_significance(
    summary["zscore"], z_cutoff, z_uc_cutoff), cmap="coolwarm")

for ax in axs[:, 0]:
    ax.set_yticklabels(labels)
for ax in axs[-1]:
    ax.set_xticklabels(labels, rotation=90)

fig.tight_layout()

fig.text(0.04, 0.07, "(*) Z-test:", fontstyle='italic')
fig.text(0.04, 0.045, f"{z_cutoff:.2f} (corrected)", fontstyle='italic')
fig.text(0.04, 0.02, f"{z_uc_cutoff:.2f} (uncorrected)", fontstyle='italic')

fig.savefig(f"stats_{statname}.pdf")
