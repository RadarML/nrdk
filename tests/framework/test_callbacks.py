"""Tests for nrdk.framework.callbacks."""

import math
from types import SimpleNamespace
from typing import cast

import pytest
import torch
from lightning import LightningModule, Trainer
from torch import nn

from nrdk.framework.callbacks import GradientStats


class _RecordingGradModule(nn.Module):
    """`pl_module` stand-in with real parameters and a recording `log_dict`."""

    def __init__(self, n_params: int = 1) -> None:
        super().__init__()
        self.weights = nn.ParameterList(
            [nn.Parameter(torch.zeros(4)) for _ in range(n_params)])
        self.log_dict_calls: list[tuple[dict, dict]] = []

    def log_dict(self, values: dict, **kwargs) -> None:
        self.log_dict_calls.append((dict(values), kwargs))


def _backward(
    callback: GradientStats, step: int, pl_module: _RecordingGradModule
) -> None:
    """Call `on_after_backward` with a lightweight `trainer` stand-in.

    `trainer`/`pl_module` are cast to `Trainer`/`LightningModule` purely to
    satisfy static typing -- `on_after_backward`'s real signature expects
    those types, but only reads a couple of duck-typed attributes from each,
    so a lightweight stand-in is enough at runtime.
    """
    trainer = SimpleNamespace(global_step=step)
    callback.on_after_backward(
        cast(Trainer, trainer), cast(LightningModule, pl_module))


# GradientStats

def test_gradient_stats_does_not_log_before_interval_reached():
    """No stats are logged until `interval` steps have elapsed."""
    callback = GradientStats(interval=4)
    pl_module = _RecordingGradModule()

    for step in range(3):
        pl_module.weights[0].grad = torch.tensor(
            [float(step + 1), 0.0, 0.0, 0.0])
        _backward(callback, step, pl_module)

    assert pl_module.log_dict_calls == []


def test_gradient_stats_logs_analytical_mean_std_min_max():
    """Logged mean/std/min/max match hand-computed values from the norms."""
    interval = 4
    callback = GradientStats(interval=interval, name="grad_norm")
    pl_module = _RecordingGradModule()

    norms = [1.0, 2.0, 3.0, 4.0]
    for step, norm in enumerate(norms):
        # A single nonzero component gives an exact L2 norm of `norm`.
        pl_module.weights[0].grad = torch.tensor([norm, 0.0, 0.0, 0.0])
        _backward(callback, step, pl_module)

    assert len(pl_module.log_dict_calls) == 1
    logged, kwargs = pl_module.log_dict_calls[0]
    assert kwargs == {"on_step": True, "sync_dist": False}

    expected_mean = sum(norms) / len(norms)
    expected_m2 = sum(n**2 for n in norms) / len(norms)
    expected_std = math.sqrt(expected_m2 - expected_mean**2)

    assert logged["grad_norm/mean"] == pytest.approx(expected_mean)
    assert logged["grad_norm/std"] == pytest.approx(expected_std)
    assert logged["grad_norm/min"] == pytest.approx(min(norms))
    assert logged["grad_norm/max"] == pytest.approx(max(norms))


def test_gradient_stats_resets_accumulators_after_logging():
    """Internal accumulators reset to their initial values after logging."""
    callback = GradientStats(interval=2)
    pl_module = _RecordingGradModule()

    for step, norm in enumerate([1.0, 2.0]):
        pl_module.weights[0].grad = torch.tensor([norm, 0.0, 0.0, 0.0])
        _backward(callback, step, pl_module)

    assert len(pl_module.log_dict_calls) == 1
    assert callback.m1 == 0.0
    assert callback.m2 == 0.0
    assert callback.min == float("inf")
    assert callback.max == 0.0


def test_gradient_stats_ignores_parameters_without_grad():
    """Parameters with `grad is None` are excluded from the norm computation."""
    callback = GradientStats(interval=1)
    pl_module = _RecordingGradModule(n_params=2)

    pl_module.weights[0].grad = torch.tensor([3.0, 4.0, 0.0, 0.0])  # norm 5
    pl_module.weights[1].grad = None

    _backward(callback, 0, pl_module)

    logged, _ = pl_module.log_dict_calls[0]
    assert logged["grad_norm/mean"] == pytest.approx(5.0)
    assert logged["grad_norm/min"] == pytest.approx(5.0)
    assert logged["grad_norm/max"] == pytest.approx(5.0)


def test_gradient_stats_combines_norms_across_multiple_parameters():
    """The total norm is computed across all parameters' grads jointly."""
    callback = GradientStats(interval=1)
    pl_module = _RecordingGradModule(n_params=2)

    pl_module.weights[0].grad = torch.tensor([3.0, 0.0, 0.0, 0.0])
    pl_module.weights[1].grad = torch.tensor([4.0, 0.0, 0.0, 0.0])

    _backward(callback, 0, pl_module)

    logged, _ = pl_module.log_dict_calls[0]
    # Concatenated vector norm: sqrt(3^2 + 4^2) == 5.
    assert logged["grad_norm/mean"] == pytest.approx(5.0)
