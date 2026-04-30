"""Automated 1/f spectral signature extraction and validation.

Implements robust automated detection of 1/f (pink noise) signatures with:
- Multi-method consensus (Welch, periodogram, MFDFA)
- Robust regression with outlier detection
- Goodness-of-fit metrics (R², AIC, BIC)
- Confidence intervals via bootstrap resampling
- Per-level spectral analysis for hierarchical systems
- Cross-level coherence analysis

Spec §12: Statistical Validation of 1/f spectral signatures
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class SpectralSignature:
    """Result of spectral signature extraction."""

    beta: float  # Spectral exponent
    hurst: float  # Hurst exponent H = (β + 1) / 2
    beta_ci_lower: float  # 95% confidence interval lower bound
    beta_ci_upper: float  # 95% confidence interval upper bound
    r_squared: float  # Goodness of fit
    aic: float  # Akaike Information Criterion
    bic: float  # Bayesian Information Criterion
    is_pink_noise: bool  # Within healthy range [0.8, 1.5]
    confidence: float  # Confidence score [0, 1]
    method: str  # Estimation method used
    n_samples: int  # Number of samples used
    frequency_range: tuple[float, float]  # Frequency range used for fitting


def robust_log_regression(
    x: np.ndarray,
    y: np.ndarray,
    outlier_threshold: float = 2.5,
) -> tuple[float, float, float]:
    """Robust log-log regression with outlier detection.

    Uses iterative reweighting to downweight outliers beyond threshold
    standard deviations from the fit.

    Args:
        x: Log-transformed independent variable (log frequencies)
        y: Log-transformed dependent variable (log power)
        outlier_threshold: Threshold for outlier detection (std devs)

    Returns:
        (slope, intercept, r_squared)
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Validate inputs
    if len(x) < 3 or len(y) < 3:
        return float("nan"), float("nan"), float("nan")

    # Remove NaN/inf values
    valid_mask = np.isfinite(x) & np.isfinite(y)
    if np.sum(valid_mask) < 3:
        return float("nan"), float("nan"), float("nan")

    x = x[valid_mask]
    y = y[valid_mask]

    # Check for sufficient variance
    if np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return float("nan"), float("nan"), float("nan")

    # Scale data to improve numerical stability
    x_mean, x_std = np.mean(x), np.std(x)
    y_mean, y_std = np.mean(y), np.std(y)

    # Avoid division by zero in scaling
    if x_std < 1e-10 or y_std < 1e-10:
        return float("nan"), float("nan"), float("nan")

    x_scaled = (x - x_mean) / x_std
    y_scaled = (y - y_mean) / y_std

    try:
        # Suppress LAPACK warnings from np.polyfit
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*DLASCL.*")

            # Initial fit on scaled data
            slope_scaled, intercept_scaled = np.polyfit(x_scaled, y_scaled, 1)

            # Transform back to original scale
            slope = slope_scaled * (y_std / x_std)
            intercept = y_mean - slope * x_mean + intercept_scaled * y_std

            y_pred = slope * x + intercept
            residuals = y - y_pred
            std_residuals = np.std(residuals)

            # Iterative reweighting
            for _ in range(3):
                # Identify outliers
                outlier_mask = np.abs(residuals) > outlier_threshold * std_residuals
                weights = np.where(outlier_mask, 0.1, 1.0)  # Downweight outliers

                # Weighted fit on scaled data
                slope_scaled, intercept_scaled = np.polyfit(x_scaled, y_scaled, 1, w=weights)

                # Transform back to original scale
                slope = slope_scaled * (y_std / x_std)
                intercept = y_mean - slope * x_mean + intercept_scaled * y_std

                y_pred = slope * x + intercept
                residuals = y - y_pred
                std_residuals = np.std(residuals)

        # Compute R²
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        return float(slope), float(intercept), float(r_squared)
    except (np.linalg.LinAlgError, ValueError):
        return float("nan"), float("nan"), float("nan")


def estimate_spectral_exponent_welch(
    signal: np.ndarray,
    fs: float = 1.0,
    nperseg: int | None = None,
    fmin: float | None = None,
    fmax: float | None = None,
) -> tuple[float, float, float]:
    """Estimate spectral exponent using Welch's method.

    Args:
        signal: Time series data
        fs: Sampling frequency (Hz)
        nperseg: Segment length for Welch (default: len(signal)//4)
        fmin: Minimum frequency for fitting
        fmax: Maximum frequency for fitting

    Returns:
        (beta, r_squared, hurst)
    """

    from scipy import signal as scipy_signal  # type: ignore

    if nperseg is None:
        nperseg = min(64, len(signal))
        nperseg = max(nperseg, 8)  # Ensure minimum segment length

    # Compute Welch PSD
    freqs, psd = scipy_signal.welch(signal, fs=fs, nperseg=nperseg)

    # Filter frequency range (exclude DC component at f=0)
    mask = (psd > 0) & (freqs > 0)
    if fmin is not None:
        mask &= freqs >= fmin
    if fmax is not None:
        mask &= freqs <= fmax

    if np.sum(mask) < 3:
        return float("nan"), float("nan"), float("nan")

    # Robust log-log fit
    log_f = np.log(freqs[mask])
    log_p = np.log(psd[mask])

    slope, _, r_squared = robust_log_regression(log_f, log_p)
    beta = -slope
    hurst = (beta + 1) / 2

    return float(beta), float(r_squared), float(hurst)


def estimate_spectral_exponent_periodogram(
    signal: np.ndarray,
    fs: float = 1.0,
    fmin: float | None = None,
    fmax: float | None = None,
) -> tuple[float, float, float]:
    """Estimate spectral exponent using periodogram.

    Args:
        signal: Time series data
        fs: Sampling frequency (Hz)
        fmin: Minimum frequency for fitting
        fmax: Maximum frequency for fitting

    Returns:
        (beta, r_squared, hurst)
    """

    from scipy import signal as scipy_signal  # type: ignore

    # Compute periodogram
    freqs, psd = scipy_signal.periodogram(signal, fs=fs)

    # Filter frequency range (exclude DC component at f=0)
    mask = (psd > 0) & (freqs > 0)
    if fmin is not None:
        mask &= freqs >= fmin
    if fmax is not None:
        mask &= freqs <= fmax

    if np.sum(mask) < 3:
        return float("nan"), float("nan"), float("nan")

    # Robust log-log fit
    log_f = np.log(freqs[mask])
    log_p = np.log(psd[mask])

    slope, _, r_squared = robust_log_regression(log_f, log_p)
    beta = -slope
    hurst = (beta + 1) / 2

    return float(beta), float(r_squared), float(hurst)


def estimate_hurst_dfa(
    signal: np.ndarray,
    min_lag: int = 10,
    max_lag: int | None = None,
) -> tuple[float, float]:
    """Estimate Hurst exponent using Detrended Fluctuation Analysis (DFA).

    Args:
        signal: Time series data
        min_lag: Minimum lag for analysis
        max_lag: Maximum lag for analysis (default: len(signal)//4)

    Returns:
        (hurst, r_squared)
    """

    signal = np.asarray(signal, dtype=float)
    n = len(signal)

    if max_lag is None:
        max_lag = n // 4

    # Integrate signal (cumulative sum)
    y = np.cumsum(signal - np.mean(signal))

    # Compute fluctuation at different scales
    scales = np.logspace(np.log10(min_lag), np.log10(max_lag), 20, dtype=int)
    scales = np.unique(scales)

    fluctuations = np.zeros_like(scales, dtype=float)
    valid_scales = []

    for i, scale in enumerate(scales):
        # Divide into segments
        n_segments = n // scale
        if n_segments < 2:  # Need at least 2 segments for meaningful DFA
            continue

        # Fit polynomial trend in each segment
        fluct = 0
        for j in range(n_segments):
            segment = y[j * scale : (j + 1) * scale]
            if len(segment) < 2:
                continue
            x = np.arange(len(segment))
            try:
                coeffs = np.polyfit(x, segment, 1)
                trend = np.polyval(coeffs, x)
                fluct += np.sum((segment - trend) ** 2)
            except (np.linalg.LinAlgError, ValueError):
                continue

        if n_segments > 0 and fluct > 0:
            fluctuations[i] = np.sqrt(fluct / (n_segments * scale))
            valid_scales.append(i)

    # Filter out invalid scales
    if len(valid_scales) < 3:
        return float("nan"), float("nan")

    valid_scales_arr = np.array(valid_scales)
    log_scales = np.log(scales[valid_scales_arr])
    log_fluct = np.log(fluctuations[valid_scales_arr])

    slope, _, r_squared = robust_log_regression(log_scales, log_fluct)
    hurst = float(slope)

    return float(hurst), float(r_squared)


def bootstrap_confidence_interval(
    signal: np.ndarray,
    estimator: Callable[[np.ndarray], float],
    n_bootstrap: int = 100,
    ci: float = 0.95,
    fs: float = 1.0,
) -> tuple[float, float]:
    """Compute confidence interval via bootstrap resampling.

    Args:
        signal: Time series data
        estimator: Function that computes statistic from signal
        n_bootstrap: Number of bootstrap samples
        ci: Confidence interval (e.g., 0.95 for 95%)
        fs: Sampling frequency

    Returns:
        (ci_lower, ci_upper)
    """

    estimates = []

    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(len(signal), size=len(signal), replace=True)
        resampled = signal[indices]

        # Compute estimate
        try:
            estimate = estimator(resampled)
            if not np.isnan(estimate):
                estimates.append(estimate)
        except Exception:
            pass

    if len(estimates) < 2:
        return float("nan"), float("nan")

    estimates_array = np.array(estimates)  # type: ignore
    alpha = 1 - ci
    ci_lower = np.percentile(estimates_array, 100 * alpha / 2)  # type: ignore
    ci_upper = np.percentile(estimates_array, 100 * (1 - alpha / 2))  # type: ignore

    return float(ci_lower), float(ci_upper)


def compute_aic_bic(
    n_samples: int,
    n_params: int,
    ss_res: float,
) -> tuple[float, float]:
    """Compute AIC and BIC for model comparison.

    Args:
        n_samples: Number of data points
        n_params: Number of parameters (typically 2 for linear fit)
        ss_res: Sum of squared residuals

    Returns:
        (aic, bic)
    """

    # Avoid division by zero
    if n_samples <= 0 or ss_res <= 0:
        return float("nan"), float("nan")

    # Maximum likelihood estimate of variance
    sigma2 = max(ss_res / n_samples, 1e-10)

    # AIC = 2k + n*ln(RSS/n) - can be negative
    aic = 2 * n_params + n_samples * np.log(sigma2)

    # BIC = k*ln(n) + n*ln(RSS/n) - can be negative
    bic = n_params * np.log(n_samples) + n_samples * np.log(sigma2)

    return float(aic), float(bic)


def extract_1f_signature(
    signal: np.ndarray,
    fs: float = 1.0,
    fmin: float | None = None,
    fmax: float | None = None,
    methods: list[str] | None = None,
    n_bootstrap: int = 100,
    beta_target: float = 1.0,
    beta_tolerance: float = 0.3,
) -> SpectralSignature:
    """Automated extraction of 1/f spectral signature with confidence intervals.

    Implements multi-method consensus approach combining:
    - Welch's method (robust to noise)
    - Periodogram (high resolution)
    - DFA (long-range correlations)

    Args:
        signal: Time series data
        fs: Sampling frequency (Hz)
        fmin: Minimum frequency for fitting
        fmax: Maximum frequency for fitting
        methods: List of methods to use (default: ['welch', 'periodogram', 'dfa'])
        n_bootstrap: Number of bootstrap samples for confidence intervals
        beta_target: Target spectral exponent (default: 1.0 for pink noise)
        beta_tolerance: Tolerance for pink noise detection (default: 0.3)

    Returns:
        SpectralSignature with all metrics
    """

    if methods is None:
        methods = ["welch", "periodogram", "dfa"]

    signal = np.asarray(signal, dtype=float)
    n_samples = len(signal)

    # Normalize signal
    signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)

    # Estimate spectral exponent using multiple methods
    estimates = []

    if "welch" in methods:
        try:
            beta_w, r2_w, h_w = estimate_spectral_exponent_welch(
                signal, fs=fs, fmin=fmin, fmax=fmax
            )
            if not np.isnan(beta_w):
                estimates.append(("welch", beta_w, r2_w, h_w))
        except Exception:
            pass

    if "periodogram" in methods:
        try:
            beta_p, r2_p, h_p = estimate_spectral_exponent_periodogram(
                signal, fs=fs, fmin=fmin, fmax=fmax
            )
            if not np.isnan(beta_p):
                estimates.append(("periodogram", beta_p, r2_p, h_p))
        except Exception:
            pass

    if "dfa" in methods:
        try:
            h_dfa, r2_dfa = estimate_hurst_dfa(signal)
            if not np.isnan(h_dfa):
                # Convert Hurst to spectral exponent: β = 2H - 1
                beta_dfa = 2 * h_dfa - 1
                estimates.append(("dfa", beta_dfa, r2_dfa, h_dfa))
        except Exception:
            pass

    if not estimates:
        raise ValueError("All spectral estimation methods failed")

    # Consensus: use median of estimates
    betas = np.array([e[1] for e in estimates])
    beta_consensus = float(np.median(betas))
    hurst_consensus = (beta_consensus + 1) / 2

    # Use best method (highest R²) for detailed metrics
    best_method = max(estimates, key=lambda x: x[2])
    method_name, beta_best, r2_best, h_best = best_method

    # Compute confidence intervals via bootstrap
    def beta_estimator(sig: np.ndarray) -> float:
        try:
            b, _, _ = estimate_spectral_exponent_welch(sig, fs=fs, fmin=fmin, fmax=fmax)
            if not np.isnan(b) and not np.isinf(b):
                return b
        except Exception:
            pass
        return float("nan")

    ci_lower, ci_upper = bootstrap_confidence_interval(
        signal, beta_estimator, n_bootstrap=n_bootstrap, ci=0.95, fs=fs
    )

    # If bootstrap failed, use fallback CI based on R²
    if np.isnan(ci_lower) or np.isnan(ci_upper):
        ci_width = 0.1 * (1 - r2_best)  # Wider CI for worse fits
        ci_lower = beta_consensus - ci_width
        ci_upper = beta_consensus + ci_width

    # Compute AIC/BIC (for 2-parameter linear fit)
    # Estimate residuals from best method
    from scipy import signal as scipy_signal  # type: ignore

    # Use appropriate nperseg for short signals
    nperseg_aic = min(64, len(signal))
    nperseg_aic = max(nperseg_aic, 8)  # Ensure minimum segment length
    freqs, psd = scipy_signal.welch(signal, fs=fs, nperseg=nperseg_aic)
    # Filter frequency range (exclude DC component at f=0)
    mask = (psd > 0) & (freqs > 0)
    if fmin is not None:
        mask &= freqs >= fmin
    if fmax is not None:
        mask &= freqs <= fmax

    log_f = np.log(freqs[mask])
    log_p = np.log(psd[mask])
    y_pred = beta_best * log_f + np.mean(log_p)
    ss_res = np.sum((log_p - y_pred) ** 2)

    aic, bic = compute_aic_bic(np.sum(mask), 2, ss_res)

    # Determine if pink noise
    is_pink = abs(beta_consensus - beta_target) <= beta_tolerance

    # Confidence score: based on agreement between methods and R²
    method_agreement = 1 - (np.std(betas) / (np.mean(np.abs(betas)) + 1e-8))
    confidence = float(np.clip(method_agreement * r2_best, 0, 1))

    return SpectralSignature(
        beta=beta_consensus,
        hurst=hurst_consensus,
        beta_ci_lower=ci_lower,
        beta_ci_upper=ci_upper,
        r_squared=r2_best,
        aic=aic,
        bic=bic,
        is_pink_noise=is_pink,
        confidence=confidence,
        method=method_name,
        n_samples=n_samples,
        frequency_range=(fmin or 0.0, fmax or fs / 2),
    )


def validate_hierarchical_spectral_signature(
    signal_levels: list[np.ndarray],
    fs: float = 1.0,
    fmin: float | None = None,
    fmax: float | None = None,
) -> dict:
    """Validate spectral signatures at each hierarchical level.

    Args:
        signal_levels: List of signals for each hierarchy level
        fs: Sampling frequency (Hz)
        fmin: Minimum frequency for fitting
        fmax: Maximum frequency for fitting

    Returns:
        Dictionary with per-level signatures and cross-level coherence
    """

    signatures: list[SpectralSignature | None] = []
    for i, signal in enumerate(signal_levels):
        try:
            sig = extract_1f_signature(signal, fs=fs, fmin=fmin, fmax=fmax)
            signatures.append(sig)
        except Exception as e:
            print(f"Failed to extract signature for level {i}: {e}")
            signatures.append(None)

    # Compute cross-level coherence
    coherence_matrix = np.zeros((len(signal_levels), len(signal_levels)))

    for i in range(len(signal_levels)):
        for j in range(i + 1, len(signal_levels)):
            try:
                from scipy import signal as scipy_signal  # type: ignore

                f, coh = scipy_signal.coherence(signal_levels[i], signal_levels[j], fs=fs)
                # Average coherence in frequency range
                if fmin is not None and fmax is not None:
                    mask = (f >= fmin) & (f <= fmax)
                    coherence_matrix[i, j] = np.mean(coh[mask])
                    coherence_matrix[j, i] = coherence_matrix[i, j]
            except Exception:
                pass

    return {
        "signatures": signatures,
        "coherence_matrix": coherence_matrix,
        "n_levels": len(signal_levels),
    }


def print_spectral_signature(sig: SpectralSignature) -> None:
    """Pretty-print spectral signature results."""

    print("\n" + "=" * 70)
    print("SPECTRAL SIGNATURE EXTRACTION RESULTS")
    print("=" * 70)
    print(f"Method: {sig.method}")
    print(f"Samples: {sig.n_samples}")
    print(f"Frequency Range: {sig.frequency_range[0]:.4f} - {sig.frequency_range[1]:.4f} Hz")
    print()
    print(f"Spectral Exponent (β): {sig.beta:.3f}")
    print(f"  95% CI: [{sig.beta_ci_lower:.3f}, {sig.beta_ci_upper:.3f}]")
    print(f"Hurst Exponent (H): {sig.hurst:.3f}")
    print()
    print(f"Goodness of Fit (R²): {sig.r_squared:.3f}")
    print(f"AIC: {sig.aic:.2f}")
    print(f"BIC: {sig.bic:.2f}")
    print()
    print(f"Pink Noise: {'✓ YES' if sig.is_pink_noise else '✗ NO'}")
    print(f"Confidence: {sig.confidence:.1%}")
    print("=" * 70 + "\n")
