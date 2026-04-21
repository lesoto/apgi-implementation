from __future__ import annotations

import numpy as np
from typing import Literal


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
