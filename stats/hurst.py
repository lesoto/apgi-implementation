from __future__ import annotations

from typing import Literal

import numpy as np


def estimate_spectral_beta(
    freqs: np.ndarray | list[float], power: np.ndarray | list[float]
) -> float:
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


def welch_periodogram(
    signal: np.ndarray, fs: float = 1.0, nperseg: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Compute power spectral density using Welch's method.

    Args:
        signal: Input time series
        fs: Sampling frequency
        nperseg: Length of each segment (default: min(256, len(signal)//4))

    Returns:
        frequencies, power spectral density
    """

    from scipy import signal as scipy_signal  # type: ignore[import-untyped]

    if nperseg is None:
        nperseg = min(256, len(signal) // 4)
    freqs, psd = scipy_signal.welch(signal, fs=fs, nperseg=nperseg, window="hann")
    return freqs, psd


def estimate_beta_welch(
    signal: np.ndarray,
    fs: float = 1.0,
    fmin: float | None = None,
    fmax: float | None = None,
) -> float:
    """Estimate β using Welch periodogram with optional frequency band selection.

    Args:
        signal: Input time series
        fs: Sampling frequency
        fmin: Minimum frequency for fit (default: fs/len(signal))
        fmax: Maximum frequency for fit (default: fs/2)

    Returns:
        Spectral exponent β where P(f) ∝ 1/f^β
    """

    freqs, power = welch_periodogram(signal, fs)
    # Select frequency band
    if fmin is None:
        fmin = freqs[1] if len(freqs) > 1 else freqs[0]
    if fmax is None:
        fmax = freqs[len(freqs) // 2]  # Use lower half to avoid high-freq noise
    mask = (freqs >= fmin) & (freqs <= fmax) & (power > 0) & (freqs > 0)
    if np.sum(mask) < 2:
        raise ValueError(f"need at least 2 frequency points in band [{fmin}, {fmax}]")
    return estimate_spectral_beta(freqs[mask], power[mask])


def hurst_from_slope(beta_spec: float) -> float:
    """H ≈ (β + 1)/2."""

    return float((beta_spec + 1.0) / 2.0)


def power_spectrum(
    freqs: np.ndarray,
    tau_levels: np.ndarray,
    sigma_levels: np.ndarray,
) -> np.ndarray:
    """Analytic multi-timescale PSD: S(f) = Σ_l σ_l²τ_l² / (1 + (2πfτ_l)²).

    Gives the closed-form power spectral density of a superposition of
    first-order Ornstein-Uhlenbeck processes with timescales τ_l and
    noise amplitudes σ_l.
    """

    f = np.asarray(freqs, dtype=float)
    taus = np.asarray(tau_levels, dtype=float)
    sigmas = np.asarray(sigma_levels, dtype=float)
    if len(taus) != len(sigmas):
        raise ValueError("tau_levels and sigma_levels must have the same length")
    S = np.zeros_like(f)
    for tau, sigma in zip(taus, sigmas):
        S += (sigma**2 * tau**2) / (1.0 + (2.0 * np.pi * f * tau) ** 2)
    return S


def dfa_analysis(
    signal: np.ndarray,
    scales: np.ndarray | list[int] | None = None,
    order: int = 1,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Detrended Fluctuation Analysis (DFA) for Hurst exponent estimation.

    Implements the DFA algorithm (Peng et al., 1994) to detect long-range
    temporal correlations in time series. More robust than spectral methods
    for non-stationary signals (e.g. APGI threshold dynamics).

    The DFA exponent α equals the Hurst exponent H for 0 < H < 1.
    Spec §22: APGI predicts H ≈ 0.8–1.1 in coupled threshold dynamics.

    Algorithm:
    1. Compute integrated profile: y(t) = Σ_{k=1}^{t} (x_k − <x>)
    2. Divide y into non-overlapping windows of size n
    3. Fit polynomial trend of degree `order` in each window and compute
       root-mean-square residual F(n)
    4. Repeat for a range of scales n
    5. α = slope of log F(n) vs log n (power-law scaling region)

    Args:
        signal: Input time series (≥ 16 samples recommended)
        scales: Window sizes to use. Defaults to 20 log-spaced values
                spanning [4, N//4].
        order: Polynomial detrending order (1 = linear, 2 = quadratic).
               Higher orders remove slower non-stationarities.

    Returns:
        (alpha, scales_used, F_values) where
        - alpha: DFA scaling exponent (≈ H for stationary processes)
        - scales_used: Array of window sizes actually evaluated
        - F_values: Corresponding fluctuation function F(n)

    Raises:
        ValueError: If signal is too short or fewer than 2 valid scales exist.
    """
    x = np.asarray(signal, dtype=float)
    N = len(x)
    if N < 16:
        raise ValueError(f"signal too short for DFA: {N} samples (need ≥ 16)")

    # Integrated profile (zero-mean detrended cumulative sum)
    y = np.cumsum(x - np.mean(x))

    # Default log-spaced scales from 4 to N//4
    if scales is None:
        min_scale = 4
        max_scale = max(8, N // 4)
        scales = np.unique(
            np.round(np.logspace(np.log10(min_scale), np.log10(max_scale), 20)).astype(int)
        )
    scales = np.asarray(scales, dtype=int)
    scales = scales[(scales >= 4) & (scales <= N // 2)]

    F_values = []
    valid_scales = []

    for n in scales:
        n_windows = N // n
        y_trunc = y[: n_windows * n].reshape(n_windows, n)
        t = np.arange(n, dtype=float)
        # Fit polynomial trend and accumulate squared residuals
        rms_sq = 0.0
        for window in y_trunc:
            coeffs = np.polyfit(t, window, order)
            trend = np.polyval(coeffs, t)
            rms_sq += np.mean((window - trend) ** 2)
        F_values.append(np.sqrt(rms_sq / n_windows))
        valid_scales.append(n)

    valid_scales_arr = np.array(valid_scales, dtype=int)
    F_arr = np.array(F_values, dtype=float)

    if len(valid_scales_arr) < 2:
        raise ValueError("fewer than 2 valid scales — signal may be too short")

    # Power-law fit: log F(n) = α log n + const
    alpha, _ = np.polyfit(np.log(valid_scales_arr), np.log(F_arr), 1)
    return float(alpha), valid_scales_arr, F_arr


def estimate_hurst_dfa(
    signal: np.ndarray,
    scales: np.ndarray | list[int] | None = None,
    order: int = 1,
) -> float:
    """Estimate Hurst exponent using Detrended Fluctuation Analysis.

    Convenience wrapper around dfa_analysis() returning only H.

    Args:
        signal: Input time series
        scales: Window sizes (default: 20 log-spaced values in [4, N//4])
        order: Polynomial detrending order (default: 1 = linear DFA)

    Returns:
        Hurst exponent H (= DFA scaling exponent α)
    """
    alpha, _, _ = dfa_analysis(signal, scales=scales, order=order)
    return alpha


def estimate_hurst_robust(
    signal: np.ndarray,
    fs: float = 1.0,
    method: Literal["welch", "raw"] = "welch",
    fmin: float | None = None,
    fmax: float | None = None,
) -> float:
    """Estimate Hurst exponent using robust spectral methods.

    Args:
        signal: Input time series
        fs: Sampling frequency
        method: "welch" for Welch periodogram, "raw" for raw FFT
        fmin, fmax: Frequency band limits for fitting

    Returns:
        Hurst exponent H
    """

    if method == "welch":
        beta = estimate_beta_welch(signal, fs, fmin, fmax)
    elif method == "raw":
        # Use raw FFT (original method)
        n = len(signal)
        fft = np.fft.fft(signal)
        power = np.abs(fft[: n // 2]) ** 2
        freqs = np.fft.fftfreq(n, 1 / fs)[: n // 2]
        beta = estimate_spectral_beta(freqs, power)
    else:
        raise ValueError(f"unknown method: {method}")
    return hurst_from_slope(beta)
