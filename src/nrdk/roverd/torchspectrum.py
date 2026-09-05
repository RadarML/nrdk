"""Torch-native raw I/Q to spectrum transform."""

from collections.abc import Mapping
from typing import Any, Literal

import torch
from abstract_dataloader.spec import Transform
from einops import rearrange
from roverd import types
from torch import Tensor
from xwr import nn as xwr_nn
from xwr import rsp as xwr_rsp

from .transforms import SpectrumData


class TorchSpectrum(Transform[
    types.XWRRadarIQ[Tensor], SpectrumData[Tensor]
]):
    """Transform raw I/Q data to 4D spectrum data via FFT (torch).

    This is the torch-native counterpart to
    [`Spectrum`][nrdk.roverd.transforms.Spectrum], which operates on numpy
    arrays; unlike that transform, it also supports range-Doppler-only
    processing via `mode="rd"`.

    Augmentations:
        - `azimuth_flip`: flip along azimuth axis.
        - `doppler_flip`: flip along doppler axis.
        - `radar_scale`: radar magnitude scale factor.
        - `radar_phase`: radar phase shift.
        - `range_scale`: apply random range scale. Excess ranges are cropped;
          missing ranges are zero-filled.
        - `speed_scale`: apply random speed scale. Excess doppler bins are
          wrapped (causing ambiguous doppler velocities); missing doppler
          velocities are zero-filled.

    !!! warning

        Range/Doppler scaling and azimuth flipping augmentations are only
        allowed when using the `full` angular spectrum mode since these
        augmentations are difficult to apply correctly in the time/spatial
        domain.

    Args:
        rsp: Radar signal processing callable to apply.
        rep: real-value representation to apply to the complex spectrum.
        mode: whether to compute full 4D spectrum or range-Doppler only.
    """

    def __init__(
        self, rsp: xwr_rsp.RSP[Tensor], rep: xwr_nn.Representation,
        mode: Literal["full", "rd"] = "full"
    ) -> None:
        self.rsp = rsp
        self.rep = rep
        self.mode = mode

    def __call__(
        self, iq: types.XWRRadarIQ[Tensor], aug: Mapping[str, Any] = {}
    ) -> SpectrumData[Tensor]:
        """Process radar data.

        Args:
            iq: input data.
            aug: augmentations to apply.

        Returns:
            Computed real spectrum, with a varying number of trailing axis
                channels depending on the provided `rep`.
        """
        # flatten ...
        batch, t = iq.iq.shape[:2]
        _iq = rearrange(
            iq.iq, "batch t doppler el az rng -> (batch t) doppler el az rng")

        if self.mode == "full":
            _cplx = self.rsp(_iq)
        else:  # self.mode == "rd"
            if any(k in aug for k in [
                "azimuth_flip", "range_scale", "speed_scale"
            ]):
                raise ValueError(
                    "Range/Doppler scaling and azimuth flipping augmentations "
                    "are not allowed for RD spectrum.")
            _cplx = self.rsp.doppler_range(xwr_rsp.iq_from_iiqq(_iq))

        # ... unflatten
        cplx = rearrange(
            _cplx, "(batch t) doppler x1 x2 rng -> batch t doppler x1 x2 rng",
            batch=batch, t=t)

        real = torch.stack([
            self.rep(sample, {k: v[i].item() for k, v in aug.items()})  # type: ignore
            for i, sample in enumerate(cplx)])

        return SpectrumData(
            spectrum=real,
            timestamps=iq.timestamps,
            range_resolution=iq.range_resolution,
            doppler_resolution=iq.doppler_resolution)
