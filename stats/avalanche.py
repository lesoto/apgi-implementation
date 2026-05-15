"""Avalanche / criticality biomarkers for APGI validation (§22).

Spec §22 predicts neural avalanche size distributions should follow power-law
scaling with exponent approximately -1.5 during conscious access.

This module provides:
- Avalanche extraction from a binary activity trace (e.g., ignition events B_t)
- Power-law exponent estimation (discrete MLE; Clauset-style)
- Simple validation helper against the spec target
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def extract_avalanches(activity: np.ndarray) -> np.ndarray:
    """Extract avalanche sizes from a binary activity time series.

    Definition (standard in avalanche analyses):
    - An avalanche is a consecutive run of non-zero activity samples
      separated by at least one zero sample.
    - The avalanche "size" here is the run length (duration in samples).

    Args:
        activity: Array-like of shape (T,), where activity[t] > 0 indicates activity.

    Returns:
        1D array of avalanche sizes (integers >= 1). Empty if none found.
    """
    x = np.asarray(activity)
    if x.ndim != 1:
        raise ValueError("activity must be 1D")

    active = x > 0
    if not np.any(active):
        return np.array([], dtype=int)

    # Identify start/end boundaries of active runs
    edges = np.diff(active.astype(int))
    starts = np.where(edges == 1)[0] + 1
    ends = np.where(edges == -1)[0] + 1

    if active[0]:
        starts = np.r_[0, starts]
    if active[-1]:
        ends = np.r_[ends, len(active)]

    sizes = ends - starts
    return sizes.astype(int)


@dataclass(frozen=True)
class PowerLawFit:
    alpha: float  # p(x) ∝ x^{-alpha}
    xmin: int
    n: int


def fit_discrete_power_law_mle(
    sizes: np.ndarray,
    xmin: int = 1,
) -> PowerLawFit:
    """Estimate discrete power-law exponent via MLE.

    Uses the standard continuous approximation for the discrete MLE:
        alpha_hat = 1 + n / Σ_i ln(x_i / (xmin - 0.5))
    valid for xmin >= 1.

    Args:
        sizes: Avalanche sizes (integers >= 1)
        xmin: Lower cutoff for the fit

    Returns:
        PowerLawFit(alpha, xmin, n)
    """
    x = np.asarray(sizes).astype(float)
    if x.ndim != 1:
        raise ValueError("sizes must be 1D")
    if xmin < 1:
        raise ValueError("xmin must be >= 1")

    x = x[np.isfinite(x)]
    x = x[x >= xmin]
    if x.size == 0:
        return PowerLawFit(alpha=float("nan"), xmin=xmin, n=0)

    denom = np.sum(np.log(x / (xmin - 0.5)))
    if denom <= 0 or not np.isfinite(denom):  # pragma: no cover
        return PowerLawFit(alpha=float("nan"), xmin=xmin, n=int(x.size))  # pragma: no cover

    alpha_hat = 1.0 + float(x.size) / float(denom)
    return PowerLawFit(alpha=float(alpha_hat), xmin=xmin, n=int(x.size))


def validate_avalanche_power_law(
    activity: np.ndarray,
    alpha_target: float = 1.5,
    tolerance: float = 0.3,
    xmin: int = 1,
    min_avalanches: int = 30,
) -> dict:
    """Validate avalanche size distribution exponent against spec target.

    Args:
        activity: Binary or non-negative activity trace (e.g., ignition B_t)
        alpha_target: Target exponent (spec: ~1.5)
        tolerance: Acceptable |alpha - target|
        xmin: Lower cutoff for fitting
        min_avalanches: Require at least this many avalanches for validation

    Returns:
        Dict with sizes, alpha, and pass/fail indicator.
    """
    sizes = extract_avalanches(activity)
    if sizes.size < min_avalanches:
        return {
            "status": "insufficient_data",
            "n_avalanches": int(sizes.size),
            "alpha": float("nan"),
            "alpha_target": float(alpha_target),
            "tolerance": float(tolerance),
            "within_tolerance": False,
            "xmin": int(xmin),
        }

    fit = fit_discrete_power_law_mle(sizes, xmin=xmin)
    within = bool(np.isfinite(fit.alpha) and abs(fit.alpha - alpha_target) <= tolerance)
    return {
        "status": "success",
        "n_avalanches": fit.n,
        "alpha": float(fit.alpha),
        "alpha_target": float(alpha_target),
        "tolerance": float(tolerance),
        "within_tolerance": within,
        "xmin": int(fit.xmin),
    }
