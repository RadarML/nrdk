"""Data augmentation specifications."""

import numpy as np


class Bernoulli:
    """Enable augmentation with certain probability.

    Args:
        p: probability of enabling (`True`).
    """

    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self) -> bool:
        return np.random.random() < self.p


class TruncatedLogNormal:
    """Truncated log-normal distribution.

    The underlying normal is always centered around zero.

    Args:
        p: probability of enabling this augmentation (`True`)
        std: standard deviation of the underlying normal distribution
        clip: clip to this many standard deviations
    """

    def __init__(
        self, p: float = 0.5, std: float = 0.2, clip: float = 2.0
    ) -> None:
        self.p = p
        self.std = std
        self.clip = clip

    def __call__(self) -> float:
        if np.random.random() > self.p:
            return 1.0

        return np.std(
            np.clip(np.random.normal(), -self.clip, self.clip) * self.std)


class Uniform:
    """Uniform distribution.

    Args:
        p: probability of enabling this augmentation.
        lower, upper: uniform bounds.
    """

    def __init__(
        self, p: float = 0.5, lower: float = -np.pi, upper: float = np.pi
    ) -> None:
        self.p = p
        self.lower = lower
        self.upper = upper

    def __call__(self) -> float:
        if np.random.random() > self.p:
            return 0.0

        return np.random.uniform(self.lower, self.upper)
