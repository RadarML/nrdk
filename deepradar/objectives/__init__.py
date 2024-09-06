"""Learning objectives for a multimodal model.

Objectives are based on :py:class:`.Objective`, and must implement
:py:meth:`.Objective.metrics`, which computes the loss (as well as any metrics
to log), organized as a :py:class:`.Metrics`:

- `Metrics.loss` should have any weighting applied (configured via class
  constructor), so that the multi-objective loss is the sum of the returned
  `Metrics.loss`.
- `Metrics.metrics` contains any additional desired metrics.

  - Our convention is that metrics should be named `{objective}_{metric}`, e.g.
    `bev_chamfer` for `chamfer` loss in the `bev` objective, and that each
    metric returns the unweighted loss `{objective}_loss`.
  - The user is responsible for ensuring that metric names do not conflict.
    Metrics must not be named `loss`, since this name is used for the global
    multi-objective loss.

- The `reduce` parameter should be respected. Objectives are expected to
  `reduce` their loss and metrics in order to allow for optimizations and
  fusing where practical.
- Use `train` to optionally exclude certain metrics from computation at train
  time.

Objectives can also optionally implement :py:meth:`.Objective.visualizations`,
which returns a dict of generated images.

- Note that :py:meth:`.Objective.visualizations` takes a dict of `Tensor` as
  input, and returns a dict of `np.ndarray`. This means the objective is
  responsible for converting `.detach()`, `.cpu()`, `.numpy()` as appropriate.
"""


from .base import Metrics, MetricValue, Objective
from .depth import Depth
from .occupancy import BEVOccupancy
from .odometry import Velocity
from .semantics import Segmentation

__all__ = [
    "MetricValue", "Metrics", "Objective",
    "BEVOccupancy", "Depth", "Segmentation", "Velocity"
]
