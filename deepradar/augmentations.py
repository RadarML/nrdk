"""Data augmentation specifications."""

import numpy as np


class Bernoulli:
    """Enable augmentation with certain probability.

    Type: `bool` (`True` if enabled).

    Args:
        p: probability of enabling.
    """

    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self) -> bool:
        return np.random.random() < self.p


class TruncatedLogNormal:
    """Truncated log-normal distribution.

    The underlying normal is always centered around zero.

    Type: `float`; returns `1.0` if not enabled.

    Args:
        p: probability of enabling this augmentation (`True`).
        std: standard deviation of the underlying normal distribution.
        clip: clip to this many standard deviations; don't clip if 0.
    """

    def __init__(
        self, p: float = 1.0, std: float = 0.2, clip: float = 2.0
    ) -> None:
        self.p = p
        self.std = std
        self.clip = clip

    def __call__(self) -> float:
        if np.random.random() > self.p:
            return 1.0

        z = np.random.normal()
        if self.clip > 0:
            z = np.clip(z, -self.clip, self.clip)
        return np.exp(z * self.std)


class Uniform:
    """Uniform distribution.

    Type: `float`; returns `0.0` if not enabled.

    Args:
        p: probability of enabling this augmentation.
        lower, upper: uniform bounds.
    """

    def __init__(
        self, p: float = 1.0, lower: float = -np.pi, upper: float = np.pi
    ) -> None:
        self.p = p
        self.lower = lower
        self.upper = upper

    def __call__(self) -> float:
        if np.random.random() > self.p:
            return 0.0

        return np.random.uniform(self.lower, self.upper)
