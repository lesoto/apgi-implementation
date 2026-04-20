from __future__ import annotations

import numpy as np


def build_timescales(tau0: float, k: float, n_levels: int) -> np.ndarray:
    """τ_i = τ0 * k^i with recommended k in [1.3, 2.0]."""

    if tau0 <= 0:
        raise ValueError("tau0 must be > 0")
    if k <= 1:
        raise ValueError("k must be > 1")
    if n_levels <= 0:
        raise ValueError("n_levels must be > 0")
    return np.array([tau0 * (k**i) for i in range(n_levels)], dtype=float)


def update_multiscale_feature(phi_prev: float, z_t: float, tau_i: float) -> float:
    """Φ_i(t+1) = (1-1/τ_i)Φ_i(t) + (1/τ_i) z(t)."""

    if tau_i <= 0:
        raise ValueError("tau_i must be > 0")
    a = 1.0 / tau_i
    return float((1.0 - a) * phi_prev + a * z_t)


def multiscale_weights(n_levels: int, k: float) -> np.ndarray:
    """w_i ∝ k^{-i}, normalized by Z."""

    raw = np.array([k ** (-i) for i in range(n_levels)], dtype=float)
    Z = np.sum(raw)
    return raw / Z


def aggregate_multiscale_signal(phi_values, pi_values, weights) -> float:
    """S = Σ_i w_i Π_i |Φ_i|."""

    phi = np.asarray(phi_values, dtype=float)
    pi = np.asarray(pi_values, dtype=float)
    w = np.asarray(weights, dtype=float)
    if not (len(phi) == len(pi) == len(w)):
        raise ValueError("phi_values, pi_values, and weights must have same length")
    return float(np.sum(w * pi * np.abs(phi)))


def apply_reset_rule(S: float, theta: float, rho: float = 0.1, delta: float = 2.0):
    return float(S * rho), float(theta + delta)
