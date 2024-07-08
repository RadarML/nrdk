"""Composable data transformations.

The following data augmentations are available and should be followed for any
new  transformations:

- `azimuth_flip`: flip along azimuth axis.
    - radar: reverse post-FFT azimuth axis.
    - lidar: reverse azimuth axis.
    - camera: flip image left-right.
    - velocity, acceleration: multiply y-component (left/right) by -1.

- `doppler_flip`: flip along doppler axis.
    - radar: reverse post-FFT doppler axis.
    - velocity, acceleration: multiply by -1.

- `range_scale`: apply random range scale.
    - radar: rescale post-FFT range axis; crop or zero-pad.
    - lidar: multiply raw ranges by scale.
    - velocity, acceleration: multiply by scale.

- `speed_scale`: apply random speed scale.
    - radar: rescale post-FFT doppler axis; wrap or zero-pad.
    - velocity, acceleration: multiple by scale.

- `radar_scale`: radar magnitude scale factor.
    - radar: multiply amplitude or complex parts.

- `radar_phase`: radar phase shift.
    - radar: add phase shift to phase component, or multiply `exp(-j * phase)`.


.. [R1] RadarHD: High resolution point clouds from mmWave Radar
    https://akarsh-prabhakara.github.io/research/radarhd/
"""

from .base import BaseTransform
from .lidar import Destagger, Map2D, DecimateMap, Depth
from .radar import (
    IIQQtoIQ, DiscardTx2, AssertTx2, FFTLinear, FFTArray,
    ComplexParts, ComplexAmplitude, ComplexPhase)


__all__ = [
    "BaseTransform",
    "Destagger", "Map2D", "DecimateMap", "Depth",
    "IIQQtoIQ", "DiscardTx2", "AssertTx2", "FFTLinear", "FFTArray",
    "ComplexParts", "ComplexAmplitude", "ComplexPhase"
]
