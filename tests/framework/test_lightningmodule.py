"""Tests for nrdk.framework.lightningmodule."""

from functools import partial

import numpy as np
import pytest
import torch
from torch import nn

from nrdk.framework.lightningmodule import NRDKLightningModule


class _TinyModel(nn.Module):
    """Small real `nn.Module`: scales an input tensor by a parameter."""

    def __init__(self, scale: float = 2.0) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale))

    def forward(self, y_true: dict) -> dict:
        return {"pred": y_true["x"] * self.scale}


class _NamedContainerModel(nn.Module):
    """Model with two named submodules, for `load_weights` rename tests."""

    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(2, 2, bias=False)
        self.b = nn.Linear(2, 2, bias=False)


class _StubObjective:
    """Structurally matches the `Objective` protocol used by the module."""

    def __init__(self, visualizations=None):
        self._visualizations = visualizations or {}
        self.render_calls: list = []
        self.call_args: list = []

    def __call__(self, y_true, y_pred, train: bool = True):
        diff = (y_pred["pred"] - y_true["x"]) ** 2
        self.call_args.append((y_true, y_pred, train))
        return diff, {"mse": diff.detach()}

    def visualizations(self, y_true, y_pred):
        return self._visualizations

    def render(self, y_true, y_pred, render_gt: bool = False):
        self.render_calls.append((y_true, y_pred, render_gt))
        return {"rendered": y_pred["pred"].detach().cpu().numpy()}


class _StubPipeline:
    """Structurally matches `abstract_dataloader.spec.Pipeline`."""

    def __init__(self):
        self.batch_calls: list = []

    def sample(self, data):
        return data

    def collate(self, data):
        return data

    def batch(self, data):
        self.batch_calls.append(data)
        return {**data, "pipeline_touched": True}


class _StubTransform:
    """Structurally matches `abstract_dataloader.spec.Transform`."""

    def __init__(self):
        self.calls: list = []

    def __call__(self, data):
        self.calls.append(data)
        return {**data, "transform_touched": True}


class _ListDataset(torch.utils.data.Dataset):
    """Map-style dataset where each item is already a full batch."""

    def __init__(self, batches):
        self.batches = batches

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        return self.batches[idx]


def _batch_loader(batches) -> torch.utils.data.DataLoader:
    """Wrap pre-made batches in a real `DataLoader`.

    `evaluate`'s `dataset` parameter is strictly type-checked against
    `DataLoader`. `batch_size=None` disables automatic collation, so each
    dataset item is yielded through unchanged.
    """
    return torch.utils.data.DataLoader(
        _ListDataset(batches), batch_size=None, shuffle=False)


def _make_module(
    model=None, objective=None, optimizer=None, transforms=None,
    vis_interval: int = 0, vis_samples: int = 16
) -> NRDKLightningModule:
    if model is None:
        model = _TinyModel()
    if objective is None:
        objective = _StubObjective()
    if optimizer is None:
        optimizer = partial(torch.optim.SGD, lr=0.1)
    return NRDKLightningModule(
        model=model, objective=objective, optimizer=optimizer,
        transforms=transforms, vis_interval=vis_interval,
        vis_samples=vis_samples)


# transform() dispatch

def test_transform_passthrough_when_none():
    """With no `transforms` configured, the batch is returned unchanged."""
    module = _make_module(transforms=None)
    batch = {"x": torch.tensor([1.0, 2.0])}
    result = module.transform(batch)
    assert result is batch


def test_transform_dispatches_to_pipeline_batch():
    """A `Pipeline`-like `transforms` object is routed through `.batch()`."""
    pipeline = _StubPipeline()
    module = _make_module(transforms=pipeline)
    batch = {"x": torch.tensor([1.0, 2.0])}
    result = module.transform(batch)

    assert len(pipeline.batch_calls) == 1
    assert result["pipeline_touched"] is True


def test_transform_dispatches_to_transform_call():
    """A `Transform`-like `transforms` object is invoked directly."""
    transform = _StubTransform()
    module = _make_module(transforms=transform)
    batch = {"x": torch.tensor([1.0, 2.0])}
    result = module.transform(batch)

    assert len(transform.calls) == 1
    assert result["transform_touched"] is True


def test_transform_moves_to_device_before_transforming():
    """The batch is moved to the given device before the transform runs."""
    transform = _StubTransform()
    module = _make_module(transforms=transform)
    batch = {"x": torch.tensor([1.0, 2.0])}
    module.transform(batch, device=torch.device("cpu"))

    moved = transform.calls[0]
    assert moved["x"].device == torch.device("cpu")
    assert torch.equal(moved["x"], batch["x"])


# load_weights

@pytest.mark.parametrize("wrap", [
    lambda w: w,
    lambda w: {"state_dict": {"model": w}},
    lambda w: {f"model.{k}": v for k, v in w.items()},
    lambda w: {f"_orig_mod.{k}": v for k, v in w.items()},
], ids=["flat", "state_dict_model_wrapper", "model_prefix", "orig_mod_prefix"])
def test_load_weights_handles_common_checkpoint_formats(tmp_path, wrap):
    """A flat, `state_dict.model`-nested, or prefixed checkpoint all load.

    Covers `model.`/`_orig_mod.`-prefixed keys, with no missing/unexpected
    keys reported in any case.
    """
    module = _make_module(model=_NamedContainerModel())
    weights = module.model.state_dict()
    path = tmp_path / "weights.pth"
    torch.save(wrap(weights), path)

    missing, unexpected = module.load_weights(str(path))
    assert missing == []
    assert unexpected == []


def test_load_weights_rename_removes_key_when_substitution_is_none(tmp_path):
    """A `rename` entry mapping to `None` drops the matching key entirely."""
    module = _make_module(model=_NamedContainerModel())
    weights = module.model.state_dict()
    path = tmp_path / "weights.pth"
    torch.save(weights, path)

    missing, unexpected = module.load_weights(
        str(path), rename=[{r"^a\.": None}])

    assert set(missing) == {"a.weight"}
    assert unexpected == []


def test_load_weights_rename_chaining_edge_case(tmp_path):
    """Later `rename` substitutions can affect the results of earlier ones.

    With `rename = [{"a": "b"}, {"b": "c"}]`, both `a.weight` and `b.weight`
    end up mapping to `c.weight` (as documented in the `load_weights`
    docstring's danger note): the first pass sends both keys to `b.weight`
    (a collision, where one silently overwrites the other), and the second
    pass then renames that surviving `b.weight` entry to `c.weight`.
    """
    module = _make_module(model=_NamedContainerModel())
    weights = {"a.weight": torch.ones(2, 2), "b.weight": torch.zeros(2, 2)}
    path = tmp_path / "weights.pth"
    torch.save(weights, path)

    missing, unexpected = module.load_weights(
        str(path), rename=[{"a": "b"}, {"b": "c"}])

    # Neither `a.weight` nor `b.weight` is present in the model any more.
    assert set(missing) == {"a.weight", "b.weight"}
    # Only a single collided/renamed `c.weight` key remains, and the model
    # has no such parameter.
    assert set(unexpected) == {"c.weight"}


# configure_optimizers

def test_configure_optimizers_returns_bound_optimizer():
    """The configured optimizer factory is bound to all model parameters."""
    module = _make_module(optimizer=partial(torch.optim.SGD, lr=0.25))
    optimizer = module.configure_optimizers()

    assert isinstance(optimizer, torch.optim.SGD)
    assert optimizer.defaults["lr"] == pytest.approx(0.25)
    n_params = sum(1 for _ in module.parameters())
    n_opt_params = sum(len(g["params"]) for g in optimizer.param_groups)
    assert n_opt_params == n_params


# on_before_optimizer_step

@pytest.mark.parametrize("finite", [True, False])
def test_on_before_optimizer_step_skips_and_zeros_on_nonfinite_grad(finite):
    """A finite gradient is left untouched.

    A non-finite one logs a skip event and zeros every gradient.
    """
    module = _make_module()
    log_calls = []
    module.log = lambda *a, **k: log_calls.append((a, k))

    params = list(module.model.parameters())
    for p in params:
        p.grad = torch.ones_like(p)
    if not finite:
        params[0].grad = torch.full_like(params[0], float("nan"))

    # `on_before_optimizer_step`'s `optimizer` argument is strictly type
    # checked against `torch.optim.Optimizer`, so a real optimizer is used
    # rather than a duck-typed stand-in.
    optimizer = torch.optim.SGD(params, lr=0.01)
    module.on_before_optimizer_step(optimizer)

    if finite:
        assert log_calls == []
        for p in params:
            assert torch.equal(p.grad, torch.ones_like(p))
    else:
        assert len(log_calls) == 1
        args, kwargs = log_calls[0]
        assert args[0] == "train/nan_grad_skip"
        assert args[1] == 1.0
        assert kwargs == {"on_step": True, "on_epoch": False}
        # `self.zero_grad()` (default `set_to_none=True`) clears all grads.
        for p in module.model.parameters():
            assert p.grad is None


# evaluate()

def test_evaluate_yields_metrics_and_rendered_outputs():
    """`evaluate` yields per-batch metrics and rendered outputs in eval mode."""
    objective = _StubObjective()
    module = _make_module(objective=objective)
    dataset = [
        {"x": torch.tensor([1.0, 2.0])},
        {"x": torch.tensor([3.0, 4.0, 5.0])},
    ]

    results = list(module.evaluate(_batch_loader(dataset), device="cpu"))

    assert len(results) == 2
    for (metrics, rendered), batch in zip(results, dataset):
        assert "loss" in metrics
        assert "mse" in metrics
        assert isinstance(metrics["loss"], np.ndarray)
        expected_pred = batch["x"].numpy() * module.model.scale.item()
        assert np.allclose(rendered["rendered"], expected_pred)
    assert len(objective.render_calls) == 2
    # `evaluate` puts the model into eval mode.
    assert module.training is False


def test_evaluate_raw_outputs_skips_render():
    """With `raw_outputs=True`, raw model outputs are returned unrendered."""
    objective = _StubObjective()
    module = _make_module(objective=objective)
    dataset = [{"x": torch.tensor([1.0, 2.0])}]

    results = list(module.evaluate(
        _batch_loader(dataset), device="cpu", raw_outputs=True))

    assert len(results) == 1
    metrics, outputs = results[0]
    assert "pred" in outputs
    expected_pred = dataset[0]["x"].numpy() * module.model.scale.item()
    assert np.allclose(outputs["pred"], expected_pred)
    assert objective.render_calls == []


def test_evaluate_applies_metadata_callable():
    """A `metadata` callable's output is merged into the yielded metrics."""
    module = _make_module()
    dataset = [{"x": torch.tensor([1.0, 2.0, 3.0])}]

    def metadata(y_true):
        return {"count": torch.tensor(float(y_true["x"].shape[0]))}

    metrics, _ = next(module.evaluate(
        _batch_loader(dataset), metadata=metadata, device="cpu"))
    assert "count" in metrics
    assert metrics["count"] == pytest.approx(3.0)
