from __future__ import annotations

from collections import deque

import numpy as np


class RunningStats:
    """Windowed running stats used for optional normalization."""

    window: deque[float]

    def __init__(self, window_size: int):
        if window_size <= 0:
            raise ValueError("window_size must be > 0")
        self.window = deque(maxlen=window_size)

    def update(self, value: float) -> None:
        self.window.append(float(value))

    def mean(self) -> float:
        return float(np.mean(self.window)) if self.window else 0.0

    def variance(self, bessel_correction: bool = True) -> float:
        """Compute variance with optional Bessel correction for unbiased estimation.

        Args:
            bessel_correction: If True, use N-1 denominator (unbiased). If False, use N (MLE).

        Returns:
            Sample variance with Bessel correction if enabled.
        """
        if not self.window:
            return 1.0
        if bessel_correction:
            return float(np.var(self.window, ddof=1))
        else:
            return float(np.var(self.window, ddof=0))

    def std(self) -> float:
        return float(np.sqrt(self.variance()))


class EMAStats:
    """Online EMA stats (Method A in §1.2).

    Implements:
    μ(t+1) = (1-α)μ(t) + α·z(t)
    σ²(t+1) = (1-α)σ²(t) + α·(z-μ)²
    """

    def __init__(self, alpha: float, initial_mean: float = 0.0, initial_var: float = 1.0):
        if not (0.0 < alpha <= 1.0):
            raise ValueError("alpha must be in (0,1]")
        self.alpha = alpha
        self._mean = initial_mean
        self._var = initial_var

    def update(self, value: float) -> None:
        # Update mean first
        self._mean = (1.0 - self.alpha) * self._mean + self.alpha * value
        # Update variance using centered deviation
        self._var = (1.0 - self.alpha) * self._var + self.alpha * (value - self._mean) ** 2

    def mean(self) -> float:
        return float(self._mean)

    def variance(self) -> float:
        return float(self._var)

    def std(self) -> float:
        return float(np.sqrt(max(self._var, 0.0)))


def compute_prediction_error(x: float, x_hat: float) -> float:
    return float(x - x_hat)


def update_prediction(x_hat: float, epsilon: float, pi: float, kappa: float) -> float:
    """Update prediction via precision-weighted gradient descent (§1.4).

    x̂(t+1) = x̂(t) + κ · Π(t) · ε(t)

    Args:
        x_hat: Current prediction
        epsilon: Prediction error
        pi: Precision weight
        kappa: Learning rate (stability: κ < 2/Π_max)

    Returns:
        Updated prediction x̂(t+1)
    """
    return float(x_hat + kappa * pi * epsilon)


def normalize_error(z: float, sigma: float, eps: float = 1e-8) -> float:
    """~z = z/(σ+ε)."""

    return float(z / (abs(sigma) + eps))


def z_score(epsilon: float, stats: RunningStats, eps: float = 1e-8) -> float:
    """Optional standardized form (ε-μ)/σ for stationary pipelines."""

    return float((epsilon - stats.mean()) / max(stats.std(), eps))
