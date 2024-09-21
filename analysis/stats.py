"""Statistical time series analysis."""

import numpy as np
from beartype.typing import NamedTuple
from jaxtyping import Float, Num, Shaped, UInt


def _pmean(x: Shaped[np.ndarray, "N"], n: int = 0) -> Float[np.ndarray, "N2"]:
    """Calculate the partial mean for the first `n` items.

    Includes the remaining elements::

        E[x[0:]], E[x[1:]], E[x[2:]], ... E[x[n:]]
    """
    nc = np.arange(x.shape[0], x.shape[0] - n, -1)
    left = np.cumsum(x[:n][::-1])[::-1]
    right = np.sum(x[n:])
    return (left + right) / nc


def autocorrelation(x: Num[np.ndarray, "N"], ) -> Float[np.ndarray, "N2"]:
    """Calculate autocorrelation for time delays up to N/2."""
    half = x.shape[0] // 2

    # First and second moments
    m1_left = _pmean(x, half)[1:]
    m1_right = _pmean(x[::-1], half)[1:]
    m2_left = _pmean(x**2, half)[1:]
    m2_right = _pmean(x[::-1]**2, half)[1:]

    # NOTE: adjust for empirical estimate of variance, covariance
    n = np.arange(x.shape[0] - 1, x.shape[0] - half, -1)
    std_left = np.sqrt((m2_left - m1_left**2) * n / (n - 1))
    std_right = np.sqrt((m2_right - m1_right**2) * n / (n - 1))
    # --> this is the only unfactorable O(n^2) step <--
    mcross = np.array([
        np.sum(x[i:] * x[:-i]) for i in range(1, half)]) / (n - 1)
    cov = (mcross - m1_left * m1_right) * n / (n - 1)

    # Estimated autocorrelation
    return cov / std_left / std_right


def effective_sample_size(x: Num[np.ndarray, "t"]) -> float:
    """Calculate effective sample size for time series data.

    Let `x` have `N` samples. For autocorrelation `rho_t`, where `t` is the
    delay, in samples, we use the estimate::

        N_eff = N / (1 + 2 * (rho_1 + rho_2 + ...))

    In our estimate, we sum up to `t = N/2` (the maximum value for which
    `rho_t` is empirically estimatable), and clip `rho_t` to positive values.
    A simplified implementation is as follows::

        rho = np.array([
            np.cov(x[i:], x[:-i])[0, 1] / np.std(x[i:]) / np.std(x[:-i])
            for i in range(1, x.shape[0] // 2)])
        return x.shape[0] / (1 + 2 * np.sum(np.maximum(0.0, rho)))

    Our implementation is optimized to reuse moment calculations and
    partial sums within moment calculations, and yields a performance
    improvement of ~20x for `N` around 2000. However, the implementation is
    still `O(N^2)` due to the autocorrelation estimate, which requires
    calculating `(x[i] * x[j])` for all `0 < i - j < N/2`.

    Args:
        x: time series data.

    Returns:
        ESS estimate.
    """
    rho = autocorrelation(x)
    rho_sum = np.sum(np.maximum(0.0, rho))
    return x.shape[0] / (1 + 2 * rho_sum)


class NDStats(NamedTuple):
    """Mean, variance, and ESS tracking.

    Attributes:
        n: number of samples.
        m1: sum of values, e.g. accumulated first moment.
        m2: sum of squares, e.g. accumulated second moment.
        ess: effective sample size estimate.
    """

    n: UInt[np.ndarray, "*size"]
    m1: Float[np.ndarray, "*shape"]
    m2: Float[np.ndarray, "*shape"]
    ess: Float[np.ndarray, "*shape"]

    @property
    def _n(self) -> Float[np.ndarray, "..."]:
        """Raw sample size, with extra dimensions to match the data."""
        n = self.n
        while len(n.shape) < len(self.m1.shape):
            n = np.expand_dims(n, -1)
        return n

    @property
    def mean(self) -> Float[np.ndarray, "*shape"]:
        """Sample mean."""
        return self.m1 / self._n

    @property
    def std(self) -> Float[np.ndarray, "*shape"]:
        """Unbiased estimate of the sample standard deviation."""
        return np.sqrt(
            (self.m2 / self._n - self.mean**2) * self._n / (self._n - 1))

    @property
    def stderr(self) -> Float[np.ndarray, "*shape"]:
        """Sample standard error, with effective sample size correction."""
        with np.errstate(invalid='ignore'):
            return self.std / np.sqrt(self.ess)

    @property
    def zscore(self) -> Float[np.ndarray, "*shape"]:
        """Z-score, assuming a zero null hypothesis."""
        return self.mean / self.stderr

    @staticmethod
    def stack(*stats) -> "NDStats":
        """Stack multiple NDStats containers."""
        try:
            return NDStats(*[np.stack(x) for x in list(zip(*stats))])
        except Exception as e:
            raise e

    def sum(self) -> "NDStats":
        """Get aggregate values."""
        if len(self.n.shape) == 0:
            raise ValueError("Can only `.sum()` a stack of statistics.")
        return NDStats(*[np.sum(x, axis=0) for x in self])
