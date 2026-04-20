from __future__ import annotations

import numpy as np


def estimate_spectral_beta(freqs, power) -> float:
    """Estimate β from P(f) ∝ 1/f^β using log-log linear fit."""

    f = np.asarray(freqs, dtype=float)
    p = np.asarray(power, dtype=float)
    mask = (f > 0) & (p > 0)
    if np.sum(mask) < 2:
        raise ValueError("need at least two positive frequency/power points")
    x = np.log(f[mask])
    y = np.log(p[mask])
    slope, _intercept = np.polyfit(x, y, 1)
    return float(-slope)


def hurst_from_slope(beta_spec: float) -> float:
    """H ≈ (β + 1)/2."""

    return float((beta_spec + 1.0) / 2.0)
