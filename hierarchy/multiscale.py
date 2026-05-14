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


def estimate_optimal_timescale_ratio(
    signal: np.ndarray,
    fs: float = 1.0,
    n_levels: int = 4,
) -> float:
    """Estimate optimal timescale ratio k from observed spectral characteristics.

    Uses spectral analysis to determine the best geometric progression ratio
    for hierarchical decomposition.

    Args:
        signal: Time series data
        fs: Sampling frequency (Hz)
        n_levels: Number of hierarchy levels

    Returns:
        Optimal timescale ratio k ∈ [1.3, 2.0]
    """

    from scipy import signal as scipy_signal  # type: ignore

    # Compute power spectrum
    freqs, psd = scipy_signal.welch(signal, fs=fs)

    # Find peaks in log-log space
    log_f = np.log(freqs[freqs > 0])
    log_p = np.log(psd[freqs > 0])

    # Smooth spectrum to find characteristic scales
    from scipy.ndimage import gaussian_filter1d  # type: ignore

    smoothed = gaussian_filter1d(log_p, sigma=2)

    # Find local maxima (characteristic timescales)
    peaks = []
    for i in range(1, len(smoothed) - 1):
        if smoothed[i] > smoothed[i - 1] and smoothed[i] > smoothed[i + 1]:
            peaks.append(log_f[i])

    if len(peaks) < 2:
        # Default to 1.6 if not enough peaks
        return 1.6

    # Estimate k from peak spacing
    peak_spacing = np.diff(peaks)
    k_estimate = np.exp(np.mean(peak_spacing))

    # Clamp to reasonable range
    k_optimal = float(np.clip(k_estimate, 1.3, 2.0))

    return k_optimal


def estimate_hierarchy_levels_from_data(
    signal: np.ndarray,
    fs: float = 1.0,
    tau_min: float = 0.01,
    tau_max: float = 10.0,
) -> int:
    """Estimate optimal number of hierarchy levels from data.

    Uses spectral analysis to determine the number of distinct timescales
    present in the signal.

    Args:
        signal: Time series data
        fs: Sampling frequency (Hz)
        tau_min: Minimum timescale (seconds)
        tau_max: Maximum timescale (seconds)

    Returns:
        Estimated number of hierarchy levels
    """

    from scipy import signal as scipy_signal  # type: ignore

    # Compute power spectrum
    freqs, psd = scipy_signal.welch(signal, fs=fs)

    # Find peaks in log-log space
    log_f = np.log(freqs[freqs > 0])
    log_p = np.log(psd[freqs > 0])

    # Smooth spectrum
    from scipy.ndimage import gaussian_filter1d  # type: ignore

    smoothed = gaussian_filter1d(log_p, sigma=2)

    # Find local maxima
    peaks = []
    for i in range(1, len(smoothed) - 1):
        if smoothed[i] > smoothed[i - 1] and smoothed[i] > smoothed[i + 1]:
            peaks.append(log_f[i])

    # Number of peaks indicates number of timescales
    n_levels = max(2, min(len(peaks), 8))

    return int(n_levels)


def update_multiscale_feature(phi_prev: float, z_t: float, tau_i: float) -> float:
    """Φ_i(t+1) = (1-1/τ_i)Φ_i(t) + (1/τ_i) z(t)."""

    if tau_i <= 0:
        raise ValueError("tau_i must be > 0")
    a = 1.0 / tau_i
    return float((1.0 - a) * phi_prev + a * z_t)


def multiscale_weights(n_levels: int, k: float) -> np.ndarray:
    """w_i ∝ k^{-i}, normalized by Z."""

    raw = np.array([k ** (-i) for i in range(n_levels)], dtype=float)
    Z = float(np.sum(raw))
    return raw / Z


def aggregate_multiscale_signal(
    phi_values: np.ndarray | list[float],
    pi_values: np.ndarray | list[float],
    weights: np.ndarray | list[float],
) -> float:
    """S = Σ_i w_i Π_i |Φ_i|."""

    phi = np.asarray(phi_values, dtype=float)
    pi = np.asarray(pi_values, dtype=float)
    w = np.asarray(weights, dtype=float)
    if not (len(phi) == len(pi) == len(w)):
        raise ValueError("phi_values, pi_values, and weights must have same length")
    return float(np.sum(w * pi * np.abs(phi)))


def aggregate_multiscale_signal_phi(
    phi_values: np.ndarray | list[float],
    pi_values: np.ndarray | list[float],
    weights: np.ndarray | list[float],
) -> float:
    """S = Σ_i w_i · Π_i · φ_i  (signed, no abs).

    For use when phi_values already hold φ(ε)-transformed errors (§12:
    S_inst⁽ˡ⁾ = Π · φ(ε) · Γ).  The sign is preserved so that aversive
    prediction errors can suppress ignition.
    """
    phi = np.asarray(phi_values, dtype=float)
    pi = np.asarray(pi_values, dtype=float)
    w = np.asarray(weights, dtype=float)
    if not (len(phi) == len(pi) == len(w)):
        raise ValueError("phi_values, pi_values, and weights must have same length")
    return float(np.sum(w * pi * phi))


def apply_reset_rule(
    S: float, theta: float, rho: float = 0.1, delta: float = 2.0
) -> tuple[float, float]:
    return float(S * rho), float(theta + delta)


def phase_signal(omega: float, t: float, phi0: float = 0.0) -> float:
    """Oscillatory phase state: ϕ_l(t) = ω_l·t + ϕ_0."""

    return float(omega * t + phi0)


def modulate_threshold(theta_0: float, pi_above: float, phi_above: float, k_down: float) -> float:
    """Phase-coupled top-down threshold modulation:
    θ_mod = θ_0 · (1 + k_down · Π_above · cos(ϕ_above)).
    """

    return float(theta_0 * (1.0 + k_down * pi_above * np.cos(phi_above)))


def bottom_up_cascade(theta: float, S_lower: float, theta_lower: float, k_up: float) -> float:
    """Hierarchical ignition suppression: θ' = θ·(1 - k_up) if S_lower > θ_lower."""

    if S_lower > theta_lower:
        return float(theta * (1.0 - k_up))
    return float(theta)


class MultiscaleWeightScheduler:
    """Adaptive per-level weight scheduler satisfying Σwₗ = 1 (spec §11).

    The spec requires level weights wₗ to:
    1. Sum to 1 (normalization — hard constraint, §11)
    2. Reflect per-level informational value from prior trials (adaptive)

    This class tracks per-level information value |φ(εₗ)| via an Exponential
    Moving Average and updates weights proportionally — the gradient-free
    Bayesian optimization approach described in §11:
    "wₗ can be initialized uniformly and updated via gradient-free Bayesian
    optimization across paradigm runs."

    Update rule (per step):
        v_ℓ(t+1) = (1 - α) · v_ℓ(t) + α · |φ(ε_ℓ)|
        w_ℓ(t+1) = v_ℓ(t+1) / Σ_j v_j(t+1)      ← guarantees Σwₗ = 1

    where α ∈ (0,1] is the EMA learning rate.
    """

    def __init__(self, n_levels: int, alpha: float = 0.05) -> None:
        """Initialize with uniform weights.

        Args:
            n_levels: Number of hierarchy levels L
            alpha: EMA learning rate α ∈ (0, 1] (default: 0.05 — slow adaptation)
        """
        if n_levels <= 0:
            raise ValueError(f"n_levels must be > 0, got {n_levels}")
        if not (0.0 < alpha <= 1.0):
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")

        self.n_levels = n_levels
        self.alpha = alpha
        self.values = np.ones(n_levels, dtype=float) / n_levels
        self.weights = np.ones(n_levels, dtype=float) / n_levels

    def update(self, phi_errors: np.ndarray | list[float]) -> np.ndarray:
        """Update weights from observed per-level φ(ε) magnitudes.

        Args:
            phi_errors: φ-transformed prediction errors per level, shape (n_levels,).
                Absolute value taken internally — information value is unsigned.

        Returns:
            Updated weight array wₗ, shape (n_levels,), guaranteed to sum to 1.

        Raises:
            ValueError: If phi_errors length does not match n_levels
        """
        phi = np.asarray(phi_errors, dtype=float)
        if len(phi) != self.n_levels:
            raise ValueError(
                f"phi_errors length {len(phi)} does not match n_levels {self.n_levels}"
            )

        self.values = (1.0 - self.alpha) * self.values + self.alpha * np.abs(phi)

        total = float(np.sum(self.values))
        if total > 1e-12:
            self.weights = self.values / total
        else:
            self.weights = np.ones(self.n_levels, dtype=float) / self.n_levels

        return self.weights.copy()

    def reset(self) -> None:
        """Reset to uniform weights (call between paradigm runs)."""
        self.values = np.ones(self.n_levels, dtype=float) / self.n_levels
        self.weights = np.ones(self.n_levels, dtype=float) / self.n_levels

    def get_weights(self) -> np.ndarray:
        """Return current weights, always summing to 1."""
        return self.weights.copy()
