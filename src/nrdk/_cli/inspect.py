"""Inspect pytorch checkpoint."""

import os
from dataclasses import dataclass

import numpy as np
import optree
import torch
from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich.tree import Tree


@dataclass
class Param:
    shape: tuple[int, ...]
    dtype: str

    def __repr__(self) -> str:
        return f"{self.dtype}[{', '.join(str(s) for s in self.shape)}]"


def _format_count(n: int) -> tuple[str, str]:
    if n > 1000000:
        size_str = f"{n / 1000000:.1f}M"
        style = "bold red"
    elif n > 1000:
        size_str = f"{n / 1000:.1f}K"
        style = "yellow"
    else:
        size_str = f"{n}"
        style = "green"
    return size_str, style


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


def _build_tree(
    tree,
    current_node: Tree,
    metadata: list[Text],
    shapes: list[Text],
    level: int = 0,
    max_levels: int = 2,
):
    for k, v in tree.items():
        node = current_node.add(k)

        n = sum(np.prod(p.shape) for p in optree.tree_leaves(v))  # type: ignore
        size_str, style = _format_count(n)
        metadata.append(Text(size_str, style=style))

        shapes.append(
            Text(str(v), style="dim white")
            if isinstance(v, Param) else Text(""))

        recurse = (
            level != max_levels
            and isinstance(v, dict)
            and len(optree.tree_leaves(v)) > 1   # type: ignore
        )
        if recurse:
            _build_tree(v, node, metadata, shapes, level + 1, max_levels)


def _get_model_path(path: str) -> str:
    if os.path.isfile(path):
        return path

    candidates = []
    for p, dirs, files in os.walk(path):
        for f in files:
            if f.split(".")[-1] in ["ckpt", "pt", "pth"]:
                candidates.append(os.path.join(p, f))

    if len(candidates) == 0:
        raise FileNotFoundError(
            f"No checkpoint (*.ckpt|pt|pth) found in {path}."
        )

    modtimes = [os.path.getmtime(c) for c in candidates]
    selected = candidates[np.argmax(modtimes)]
    print(f"(inspecting: {selected})\n")
    return selected


def cli_inspect(
    path: str, /, depth: int = 3, weights_only: bool = False
) -> None:
    """Inspect a pytorch / pytorch lightning checkpoint.

    !!! info "Usage"

        Inspect a representative (most recent) checkpoint:
        ```sh
        nrdk inspect results/experiment/version
        ```

    If the `path` points to a file, inspect that checkpoint; if it points to a
    directory, inspect the most recent checkpoint (by modification time).

    Args:
        path: path to checkpoint file.
        depth: maximum depth to print in the module/parameter tree, in layers;
            set `--depth 0` to print the full tree.
        weights_only: allow loading pytorch checkpoints containing custom
            objects. Note that this allows arbitrary code execution!
    """
    path = _get_model_path(path)
    contents = torch.load(path, map_location="cpu", weights_only=weights_only)

    if "state_dict" in contents:
        contents = contents["state_dict"]

    tree = {}
    for k, v in contents.items():
        name = k.split(".")
        subtree = tree
        for subpath in name[:-1]:
            subtree = subtree.setdefault(subpath, {})

        if isinstance(v, torch.Tensor):
            dtype = str(v.dtype).replace("torch.", "")
            dtype = dtype.replace("float", "f").replace("bfloat", "bf")
            subtree[name[-1]] = Param(shape=tuple(v.shape), dtype=dtype)

    tree = _collapse_singletons(tree)

    console = Console()
    total_params = sum(np.prod(p.shape) for p in optree.tree_leaves(tree))  # type: ignore
    size_str, style = _format_count(total_params)

    metadata = [Text(f"{size_str}", style=style)]
    shapes = [Text("", style=style)]
    root = Tree("total")
    _build_tree(tree, root, metadata, shapes, 0, depth - 1)

    table = Table(box=None, pad_edge=False, show_header=False, show_edge=False)
    table.add_row(root, Text("\n").join(metadata), Text("\n").join(shapes))
    console.print(table)
