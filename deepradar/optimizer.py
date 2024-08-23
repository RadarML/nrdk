"""Optimizer management."""

import re
from torch import nn, optim


def create_optimizer(
    module: nn.Module, name: str = "AdamW", default_lr: float = 1e-4,
    warmup: int = 100, subsets: dict = {}
):
    """Create optimizer.
    
    Args:
        module: pytorch module (or subclass, e.g. `LightningModule`) to create
            optimizer for; used for iterating through parameters.
        name: name of optimizer (in `torch.optim`).
        default_lr: optimizer base learning rate.
        warmup: warmup period, in iterations.
        subsets: subsets of parameters to use a different learning rate for;
            each key is a regex, and each value is the corresponding learning
            rate.
    
    Returns:
        See the documentation for `LightningModule.configure_optimizers`:
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#configure-optimizers
    """
    _subsets = {s: [] for s in subsets}
    nomatch = []
    for key, param in module.named_parameters():
        for pattern in subsets:
            if re.match(pattern, key):
                _subsets[pattern].append(param)
                break
        else:
            nomatch.append(param)

    opt =  getattr(optim, name)([
        {"params": _subsets[k], **subsets[k]}
        for k in _subsets
    ] + [{"params": nomatch}], lr=default_lr)

    if warmup > 0:
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": optim.lr_scheduler.LinearLR(
                    opt, start_factor=1e-3, end_factor=1.0,
                    total_iters=warmup),
                "interval": "step"}
        }
    else:
        return opt
