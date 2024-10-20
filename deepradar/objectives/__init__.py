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
  responsible for converting `.numpy()` as appropriate.
- The caller will convert `.detach()` and `.cpu()`;
  :py:meth:`.Objective.visualizations` is called asynchronously (on the first
  worker only (`global_rank == 0`) in distributed training), and should not use
  any GPU acceleration.

.. [L1] Focal Loss for Dense Object Detection
  https://arxiv.org/abs/1708.02002v2
"""


from .base import (
    LPObjective,
    Metrics,
    MetricValue,
    Objective,
    PointCloudObjective,
    accuracy_metrics,
    focal_loss_with_logits,
)
from .depth import Depth
from .occupancy import BEVOccupancy
from .occupancy3 import PolarOccupancy
from .odometry import Velocity
from .semantics import Segmentation

__all__ = [
    "LPObjective", "PointCloudObjective",
    "accuracy_metrics", "focal_loss_with_logits",
    "MetricValue", "Metrics", "Objective", "PolarOccupancy",
    "BEVOccupancy", "Depth", "Segmentation", "Velocity"
]
