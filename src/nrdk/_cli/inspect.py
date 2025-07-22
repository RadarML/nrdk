"""Inspect pytorch checkpoint."""

import os

import numpy as np
import optree
import torch


def _pc(tree):
    n = optree.tree_reduce(lambda x, y: x + y, tree, initial=0)
    if n > 1000000:
        return f"{n / 1000000:.1f}M"
    elif n > 1000:
        return f"{n / 1000:.1f}K"
    else:
        return str(n)


def _collapse_singletons(tree):
    out = {}
    for k, v in tree.items():
        if isinstance(v, dict) and len(v) == 1:
            out[f"{k}.{next(iter(v.keys()))}"] = next(iter(v.values()))
        elif isinstance(v, dict):
            out[k] = _collapse_singletons(v)
        else:
            out[k] = v
    return out

def _print_tree(tree, level: int = 0, max_levels: int = 2):
    if level == 0:
        print(f"{'total':32}{_pc(tree)}")
        print("-" * 40)

    for k, v in tree.items():
        print(f"{'  ' * level}{k}{' ' * (32 - len(k) - level * 2)}{_pc(v)}")

        recurse = (
            level != max_levels
            and isinstance(v, dict)
            and len(optree.tree_leaves(v)) > 1)  # type: ignore
        if recurse:
            _print_tree(v, level + 1, max_levels)


def _get_model_path(path: str) -> str:
    if os.path.isfile(path):
        return path

    candidates = []
    for p, dirs, files in os.walk(path):
        for f in files:
            if f.split('.')[-1] in ['ckpt', 'pt', 'pth']:
                candidates.append(os.path.join(p, f))

    if len(candidates) == 0:
        raise FileNotFoundError(
            f"No checkpoint (*.ckpt|pt|pth) found in {path}.")

    modtimes = [os.path.getmtime(c) for c in candidates]
    selected = candidates[np.argmax(modtimes)]
    print(f"(inspecting: {selected})\n")
    return selected


def cli_inspect(
    path: str, /, depth: int = 2, weights_only: bool = False
) -> None:
    """Inspect a pytorch / pytorch lightning checkpoint.

    If the `path` points to a file, inspect that checkpoint; if it points to a
    directory, inspect the most recent checkpoint (by modification time).

    Args:
        path: path to checkpoint file.
        depth: maximum depth to print in the module/parameter tree.
        weights_only: allow loading pytorch checkpoints containing custom
            objects. Note that this allows arbitrary code execution!
    """
    path = _get_model_path(path)
    contents = torch.load(path, map_location='cpu', weights_only=weights_only)

    if 'state_dict' in contents:
        contents = contents['state_dict']

    tree = {}
    for k, v in contents.items():
        name = k.split('.')
        subtree = tree
        for subpath in name[:-1]:
            subtree = subtree.setdefault(subpath, {})

        if isinstance(v, torch.Tensor):
            subtree[name[-1]] = np.prod(tuple(v.shape))

    tree = _collapse_singletons(tree)
    _print_tree(tree, 0, depth)
