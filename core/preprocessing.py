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


def compute_prediction_error(x: float, x_hat: float) -> float:
    return float(x - x_hat)


def normalize_error(z: float, sigma: float, eps: float = 1e-8) -> float:
    """~z = z/(σ+ε)."""

    return float(z / (abs(sigma) + eps))


def z_score(epsilon: float, stats: RunningStats, eps: float = 1e-8) -> float:
    """Optional standardized form (ε-μ)/σ for stationary pipelines."""

    return float((epsilon - stats.mean()) / max(stats.std(), eps))
