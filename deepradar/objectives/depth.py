"""3D depth estimation objective."""

import torch
from torch import Tensor
from torchvision.transforms import Resize, InterpolationMode

from jaxtyping import Shaped, Float

from .base import MetricValue, Module


class LPDepth:
    """Generic lp depth loss, with missing value masking.

    Args:
        ord: loss order; only l1 (`1`) and l2 (`2`) are supported.
    """

    def __init__(self, ord: int = 1) -> None:
        self.ord = ord

    def __call__(
        self, y_hat: Float[Tensor, "batch azimuth range"],
        y_true: Float[Tensor, "batch azimuth range"], reduce: bool = True
    ) -> MetricValue:

        if self.ord == 1:
            diff = torch.abs(y_hat - y_true)
        elif self.ord == 2:
            diff = y_hat * y_hat - y_true * y_true
        else:
            raise NotImplementedError("Only L1 and L2 losses are implemented.")

        mask = (y_true != 0)
        res = torch.sum(diff * mask, dim=(1, 2)) / torch.sum(mask, dim=(1, 2))
        if reduce:
            res = torch.mean(res)
        return res


class Depth(Module):
    """Radar -> lidar as depth estimation.

    Args:
        loss_order: Loss type (l1/l2).
    """

    STOPPING_METRIC = "loss/val"
    DEFAULT_CMAP = "viridis"

    def __init__(self, loss_order: int = 1, **kwargs) -> None:
        super().__init__(**kwargs)
        self.loss = LPDepth(ord=loss_order)
        self.save_hyperparameters()

    def convert_image(
        self, img: Shaped[torch.Tensor, "batch h w"]
    ) -> Shaped[torch.Tensor, "batch h2 w2"]:
        """Convert image for display."""
        return Resize(
            (512, 512 * 2), interpolation=InterpolationMode.NEAREST)(img)

    def training_step(self, batch, batch_idx):  # type: ignore
        """Standard lightning training step."""
        y_hat = self.model(batch['radar'])
        loss = self.loss(y_hat, batch['lidar'])

        self.log(
            "loss/train", loss, on_step=True, on_epoch=True, sync_dist=True)
        if self.global_step % self.log_interval == 0:
            self.log_image_comparison(
                "sample/train", batch['lidar'][:self.num_examples],
                y_hat[:self.num_examples].detach())

        return loss

    def validation_step(self, batch, batch_idx):  # type: ignore
        """Standard lightning validation step."""
        if batch_idx == 0:
            samples = self.trainer.datamodule.val_samples  # type: ignore
            radar = torch.from_numpy(
                samples['radar']).to(batch['radar'].device)
            y_true = torch.from_numpy(
                samples['lidar']).to(batch['lidar'].device)

            y_hat = self.model(radar)
            self.log_image_comparison("sample/val", y_true, y_hat)

        y_hat = self.model(batch['radar'])
        self.log("loss/val", self.loss(y_hat, batch['lidar']), sync_dist=True)

    def evaluation_step(
        self, batch
    ) -> dict[str, Shaped[torch.Tensor, "batch"]]:
        """Evaluate model with mo metric aggregation."""
        y_hat = self.model(batch['radar'])
        val_loss = self.loss(y_hat, batch['lidar'], reduce=False)
        return {"loss": val_loss}
