"""Lightning-based dataloader."""

import os
from beartype.typing import cast

import numpy as np
from torch.utils.data import Dataset
import lightning as L

import rover


class RoverDataModule(L.LightningDataModule):
    """Rover dataloader."""

    def __init__(self, datasets: list[str]) -> None:
        super().__init__()
        self.datasets = datasets


class RoverTrace:
    """Single rover trace."""

    def __init__(
        self, path: str, indices: str = "_fusion/indices.npz",
        channels: dict[str, tuple[str, str]] = {}
    ) -> None:
        npz = np.load(os.path.join(path, indices))
        self.indices = npz["indices"]
        self.channel_names = {
            n: i for i, n in enumerate(npz["sensors"])}

        self.dataset = rover.Dataset(path)
        self.channels = {
            k: self.dataset.get(v1)[v2] for k, (v1, v2) in channels.items()}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        res = {
            k: v.index(self.indices[idx, self.channel_names[k]])
            for k, v in self.channels.items()}
        if "lidar" in res:
            res["lidar"] = cast(
                rover.LidarData, self.dataset["lidar"]).destagger(res["lidar"])
            nc = res["lidar"].shape[1]
            res["lidar"] = res["lidar"][:, nc // 4:-nc // 4]
        
        return res


class RoverData(Dataset):
    """Collection of rover traces."""

    def __init__(
        self, paths: list[str], indices: str = "_fusion/indices.npz",
        channels: dict[str, tuple[str, str]] = {
            "lidar": ("_lidar", "rng"),
            "radar": ("_radar", "raw")}
    ) -> None:
        self.traces = [
            RoverTrace(p, indices=indices, channels=channels) for p in paths]

    def __len__(self):
        return sum(len(t) for t in self.traces)

    def __getitem__(self, idx):
        for trace in self.traces:
            if idx < len(trace):
                return trace[idx]
            idx -= len(trace)
        else:
            raise ValueError("Index out of bounds.")
