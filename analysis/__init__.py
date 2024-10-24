"""Result analysis utilities for models evaluated on time series data.

File Structure
--------------

All experiment results should be stored in a directory (or symlink to a
directory), nominally `results`. Each experiment folder should contain a
marker file, nominally `hparams.yaml`, and further contain evaluation traces
in a `eval` folder, potentially with arbitrary nesting::

    results/
        full_path_to/           # Arbitrary nesting is allowed
            experiment/         # Experiments may not contain other experiments
                hparams.yaml    # See `Results(marker_name=...)`
                eval/
                    ...
            ...
        ...

Within each `eval` folder, data should be stored in `.npz` files. The full path
to each trace should match between different results. For example, the
structure

::

    experiment1/
        eval/
            namespace/
                trace.npz/
                    metric.npy
    experiment2/
        eval/
            namespace/
                trace.npz/
                    metric.npy

corresponds to the evaluation trace `namespace/trace`, with the metric
`metric.npy`.

Note that these trace/metric values are expected to have the same type and
shape between different evaluations.

Usage
-----

All operations can be accessed using :py:class:`Results`:

.. code-block:: python

    results = Results(path="results")
    comparison = results.compare(
        ["experiment/a", "experiment/b", "experiment/c"], key="metric_name")

`results.compare(...)` generates a vectorized set of statistics
(:py:class:`ComparativeStats`) over all validation traces that the listed
experiments have in common. In addition to examining these stats separately,
we can aggregate them:

.. code-block:: python

    aggregate = comparison.sum()

From the aggregated (or vectorized) :py:class:`ComparativeStats`, we can easily
calculate key statistics:

- :py:meth:`ComparativeStats.percent`: percent difference between each pair of
  methods.
- :py:meth:`ComparativeStats.significance`,
  :py:meth:`ComparativeStats.z_boundary`: significance matrix and z-score
  cutoffs, for testing whether the difference between each pair of methods
  is significant.

We can also get statistics related to the underlying absolute
(:py:attr:`ComparativeStats.abs`) and relative
(:py:attr:`ComparativeStats.diff`) metric values:

- `.(abs|diff):` :py:meth:`NDStats.mean`, :py:meth:`NDStats.std`: mean and
  sample standard deviation of the metric or metric differences.
- `.(abs|diff):` :py:meth:`NDStats.stderr`: standard error with effective
  sample size correction; see :py:func:`effective_sample_size` for a detailed
  explanation.
- `.diff:` :py:meth:`NDStats.zscore`: z-score of the difference calculated
  using the standard error estimate and a null hypothesis of no difference
  between each pair of methods.
"""


from .plot import comparison_grid, comparison_matrix
from .result import ComparativeStats, Result
from .results import Results
from .stats import NDStats, autocorrelation, effective_sample_size

__all__ = [
    "comparison_matrix", "comparison_grid",
    "ComparativeStats", "Result", "Results",
    "NDStats", "autocorrelation", "effective_sample_size"
]
