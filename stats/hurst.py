from __future__ import annotations

import warnings
from typing import Literal

import numpy as np

from core.numerics import finite_mask, safe_log, safe_nperseg


class SpectralRegimeBoundaryWarning(UserWarning):
    """Emitted when β_spec sits in the fGn/fBm regime-boundary band [0.8, 1.2].

    A deliberate APGI signal, not a defect: MathSpec §12 states the H(β)
    conversion is discontinuous at β = 1.0 (form (a) gives H = 1.0, form (b)
    gives H = 0.0 — a full Hurst unit apart) and directs callers to estimate H
    directly via DFA in this band. Its own class so tests and callers can
    handle it precisely instead of blanket-silencing ``UserWarning``.
    """


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

    # Clamp rather than let scipy warn and silently rewrite nperseg, so the
    # segment length actually used is the one computed here. `len // 4` alone
    # yields 0 for very short signals, which scipy then replaces with its own
    # default of 256 — larger than the signal.
    nperseg = safe_nperseg(len(signal), nperseg)
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


def hurst_from_slope(
    beta_spec: float,
    regime: Literal["auto", "fgn", "fbm"] = "auto",
    warn_near_boundary: bool = True,
) -> float:
    """Convert spectral exponent β to Hurst exponent H.
    MathSpec §12: there are TWO regime-dependent conversion formulas with an
    explicit DISCONTINUITY at β_spec=1.0, not one universal formula:
        (a) fractional Gaussian noise (fGn): H = (β+1)/2, valid β ∈ (0,1),
            giving H ∈ (0.5, 1.0)
        (b) fractional Brownian motion (fBm): H = (β-1)/2, valid β ∈ (1,3),
            giving H ∈ (0.0, 1.0)
    At β=1.0 exactly, (a) gives H=1.0 while (b) gives H=0.0 — a full Hurst
    unit apart; these characterise different processes (fGn increment vs
    fBm integrated trajectory) and are not interchangeable. Applying the
    fGn formula unconditionally across the whole β range (the historical
    bug in this codebase) silently produces the wrong regime's H whenever
    β > 1.
    For cortical EEG data with β_spec ∈ [0.8, 1.2] (the biologically common
    1/f regime straddling the boundary), the spec says: do not convert via
    either formula — estimate H directly via DFA (see dfa_analysis /
    estimate_hurst_dfa). This function still returns a best-effort value in
    that band (regime="auto" picks fGn for β<1, fBm for β>1, fGn at exactly
    β=1) but emits a warning by default, since neither formula is reliable
    there.
    Args:
        beta_spec: Spectral exponent β (P(f) ∝ 1/f^β)
        regime: "auto" (default, selects by β<1 vs β>1), "fgn" (force
            (β+1)/2), or "fbm" (force (β-1)/2)
        warn_near_boundary: If True (default), warn when β ∈ [0.8, 1.2] and
            regime="auto", per the spec's DFA-instead-of-conversion guidance.
    Returns:
        Estimated Hurst exponent H
    """
    if regime == "fgn":
        return float((beta_spec + 1.0) / 2.0)
    if regime == "fbm":
        return float((beta_spec - 1.0) / 2.0)
    if regime != "auto":
        raise ValueError(f"unknown regime: {regime!r}, must be one of 'auto', 'fgn', 'fbm'")
    if warn_near_boundary and 0.8 <= beta_spec <= 1.2:
        warnings.warn(
            f"beta_spec={beta_spec:.3f} is within the regime-boundary band [0.8, 1.2] "
            "(MathSpec §12) — the fGn/fBm conversion is discontinuous at β=1.0 and "
            "unreliable here. Estimate H directly via DFA (dfa_analysis / "
            "estimate_hurst_dfa) instead of converting from the spectral slope.",
            SpectralRegimeBoundaryWarning,
            stacklevel=2,
        )
    if beta_spec <= 1.0:
        return float((beta_spec + 1.0) / 2.0)
    return float((beta_spec - 1.0) / 2.0)


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
    # Power-law fit: log F(n) = α log n + const.
    #
    # A perfectly constant series has F(n) = 0 at every scale, so log F = -inf.
    # Feeding that to polyfit reaches LAPACK, which prints "On entry to DLASCL,
    # parameter number 4 had an illegal value" to stderr — text no Python
    # warning filter can intercept — and returns a meaningless slope. Drop
    # non-finite points first and require two survivors.
    log_scales = safe_log(valid_scales_arr)
    log_F = safe_log(F_arr)
    mask = finite_mask(log_scales, log_F) & (F_arr > 0)
    if int(np.count_nonzero(mask)) < 2:
        raise ValueError(
            "DFA fluctuation function is zero or non-finite at nearly every scale — "
            "the series is constant (or near-constant), so no scaling exponent is "
            "defined. A constant threshold has no long-range correlation structure "
            "to estimate."
        )
    alpha, _ = np.polyfit(log_scales[mask], log_F[mask], 1)
    return float(alpha), valid_scales_arr, F_arr


#: DFA convention labels. "fgn" — stationary series (the threshold process
#: itself, an oscillation amplitude envelope): H = α_DFA. "fbm" — cumulatively
#: integrated or broadband series: H = α_DFA − 1.
Convention = Literal["fgn", "fbm"]

#: α_DFA at or above this value cannot be an fGn Hurst exponent (H ∈ (0,1)),
#: so the series is being read under the fBm convention.
FGN_FBM_BOUNDARY = 1.0


def infer_convention(alpha_dfa: float) -> Convention:
    """Infer the DFA convention from the scaling exponent's magnitude.

    Notation Appendix: "H ∈ (0, 1) by definition… literature values ≥ 1 are
    α_DFA, not H." An exponent at or above 1.0 therefore cannot be an fGn
    Hurst exponent and indicates the integrated (fBm) reading.

    Args:
        alpha_dfa: DFA scaling exponent.

    Returns:
        ``"fbm"`` when ``alpha_dfa >= 1.0``, else ``"fgn"``.
    """
    return "fbm" if alpha_dfa >= FGN_FBM_BOUNDARY else "fgn"


def hurst_from_alpha_dfa(alpha_dfa: float, convention: Convention | Literal["auto"] = "auto") -> float:
    """Convert a DFA scaling exponent to a Hurst exponent H ∈ (0, 1).

    MathSpec §12 / Notation Appendix: "For stationary processes, α_DFA = H…
    The same signal analysed cumulatively (integrated broadband or BOLD
    series) follows fBm: α_DFA = H + 1." Hence H = α_DFA (fGn) or
    H = α_DFA − 1 (fBm).

    Args:
        alpha_dfa: DFA scaling exponent.
        convention: ``"fgn"``, ``"fbm"``, or ``"auto"`` to infer from
            magnitude via :func:`infer_convention`.

    Returns:
        Hurst exponent H.

    Raises:
        ValueError: If ``convention`` is not a recognised label.
    """
    conv = infer_convention(alpha_dfa) if convention == "auto" else convention
    if conv == "fgn":
        return float(alpha_dfa)
    if conv == "fbm":
        return float(alpha_dfa - 1.0)
    raise ValueError(f"convention must be 'fgn', 'fbm', or 'auto', got {convention!r}")


def estimate_alpha_dfa(
    signal: np.ndarray,
    scales: np.ndarray | list[int] | None = None,
    order: int = 1,
) -> float:
    """Estimate the DFA scaling exponent α_DFA.

    This is the PRIMARY estimator and the quantity APGI's predictions and
    falsifiers are stated on: "Prediction and falsifier are stated on α_DFA —
    support: α_DFA ∈ 0.85–0.95; falsification: α_DFA < 0.55."

    α_DFA is NOT a Hurst exponent. It is unbounded above (a random walk gives
    α_DFA ≈ 1.5), whereas H ∈ (0, 1) by definition. Use :func:`estimate_hurst`
    when you want H, and state the convention wherever a number appears.

    Args:
        signal: Input time series.
        scales: Window sizes (default: 20 log-spaced values in [4, N//4]).
        order: Polynomial detrending order (default: 1 = linear DFA).

    Returns:
        DFA scaling exponent α_DFA.
    """
    alpha, _, _ = dfa_analysis(signal, scales=scales, order=order)
    return alpha


def estimate_hurst(
    signal: np.ndarray,
    convention: Convention | Literal["auto"] = "auto",
    scales: np.ndarray | list[int] | None = None,
    order: int = 1,
) -> float:
    """Estimate the Hurst exponent H ∈ (0, 1) via DFA, with explicit convention.

    Estimates α_DFA and converts it under the stated convention. DFA is the
    spec's preferred route to H because it is "robust to non-stationarities —
    including the slow allostatic drift in θ(t) — that invalidate spectral
    slope estimation", and because the β_spec → H conversion is discontinuous
    at β_spec = 1.

    Args:
        signal: Input time series.
        convention: ``"fgn"`` (stationary; H = α_DFA), ``"fbm"`` (integrated;
            H = α_DFA − 1), or ``"auto"`` to infer from magnitude.
        scales: Window sizes.
        order: Polynomial detrending order.

    Returns:
        Hurst exponent H.
    """
    return hurst_from_alpha_dfa(
        estimate_alpha_dfa(signal, scales=scales, order=order), convention
    )


def estimate_hurst_dfa(
    signal: np.ndarray,
    scales: np.ndarray | list[int] | None = None,
    order: int = 1,
) -> float:
    """[DEPRECATED] Alias for :func:`estimate_alpha_dfa`.

    The name is a convention collision the Notation Appendix explicitly
    forbids: this returns α_DFA, which exceeds 1 for integrated series and so
    cannot be a Hurst exponent. Use :func:`estimate_alpha_dfa` for α_DFA (the
    quantity the predictions and falsifiers are stated on) or
    :func:`estimate_hurst` for H with a declared convention.

    Args:
        signal: Input time series.
        scales: Window sizes.
        order: Polynomial detrending order.

    Returns:
        DFA scaling exponent α_DFA — NOT the Hurst exponent.
    """
    return estimate_alpha_dfa(signal, scales=scales, order=order)


def wavelet_variance_analysis(
    signal: np.ndarray,
    min_level: int = 1,
    max_level: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Haar-wavelet multiresolution variance decomposition.
    Repeatedly applies the (orthonormal) Haar wavelet transform:
        detail_j[i]   = (approx_{j-1}[2i] - approx_{j-1}[2i+1]) / sqrt(2)
        approx_j[i]   = (approx_{j-1}[2i] + approx_{j-1}[2i+1]) / sqrt(2)
    starting from approx_0 = signal, and records the variance of the detail
    coefficients at each scale j (dyadic scale 2^j). Self-contained (no
    external wavelet library dependency) since only the Haar wavelet is
    needed for the scaling-exponent estimate below.
    Args:
        signal: Input time series
        min_level: Coarsest... (lowest) level to include, default 1
        max_level: Finest limit on levels (default: floor(log2(N)) - 2, so
            each retained scale has enough coefficients for a stable variance)
    Returns:
        (levels, log2_variances) — level indices j and log2(Var(detail_j))
        for each level with at least 4 detail coefficients and nonzero variance
    """
    x = np.asarray(signal, dtype=float)
    N = len(x)
    if N < 16:
        raise ValueError(f"signal too short for wavelet analysis: {N} samples (need >= 16)")
    if max_level is None:
        max_level = max(min_level + 1, int(np.floor(np.log2(N))) - 2)
    approx = x - np.mean(x)
    levels = []
    log2_vars = []
    for j in range(1, max_level + 1):
        n = len(approx) - (len(approx) % 2)
        if n < 8:
            break
        approx = approx[:n]
        evens = approx[0::2]
        odds = approx[1::2]
        detail = (evens - odds) / np.sqrt(2.0)
        new_approx = (evens + odds) / np.sqrt(2.0)
        if j >= min_level and len(detail) >= 4:
            var_d = float(np.var(detail))
            if var_d > 0:
                levels.append(j)
                log2_vars.append(np.log2(var_d))
        approx = new_approx
    return np.array(levels, dtype=int), np.array(log2_vars, dtype=float)


def estimate_hurst_wavelet(
    signal: np.ndarray,
    min_level: int = 1,
    max_level: int | None = None,
    convention: Convention | Literal["auto"] = "auto",
) -> float:
    """Estimate the Hurst exponent via Haar-wavelet multiresolution variance
    analysis (Abry & Veitch, 1998 style log-scale-variance estimator).
    Notation Appendix: "For short series (<200 trials) or non-stationary
    processes, use wavelet-based Hurst estimation as a cross-validation" —
    wavelets localize in both time and scale, making this estimator more
    robust than global spectral-slope estimation to non-stationary drift
    (e.g. the slow allostatic drift in θ(t)) and usable on shorter series
    than DFA typically requires for stable window statistics.
    The detail-coefficient variance at dyadic scale 2^j scales as a power
    law in j for self-similar processes; log2(Var(detail_j)) vs j has slope
    β, converted to H via the same regime-aware fGn/fBm relationship used
    for spectral-slope estimates (hurst_from_slope).
    Args:
        signal: Input time series
        min_level: Coarsest wavelet level to include in the fit (default 1)
        max_level: Finest wavelet level (default: automatic)
        convention: ``"fgn"``, ``"fbm"``, or ``"auto"``. ``"auto"`` keeps the
            historical regime-detecting behaviour (fGn for slope < 1, fBm
            above); an explicit value forces that regime, which is what
            :func:`cross_validate_hurst` needs so both estimators report H in
            the same convention.
    Returns:
        Estimated Hurst exponent H ∈ (0, 1)
    Raises:
        ValueError: If the signal is too short or fewer than 2 valid scales
            are available for the slope fit.
    """
    levels, log2_vars = wavelet_variance_analysis(signal, min_level, max_level)
    if len(levels) < 2:
        raise ValueError(
            "fewer than 2 valid wavelet scales — signal may be too short "
            "for a reliable wavelet-based Hurst estimate"
        )
    slope, _ = np.polyfit(levels, log2_vars, 1)
    regime: Literal["auto", "fgn", "fbm"] = "auto" if convention == "auto" else convention
    return hurst_from_slope(float(slope), regime=regime, warn_near_boundary=False)


def cross_validate_hurst(
    signal: np.ndarray,
    agreement_tolerance: float = 0.15,
    convention: Convention | Literal["auto"] = "auto",
) -> dict:
    """Cross-validate H from DFA against the wavelet estimator.

    Notation Appendix: "For short series (<200 trials) or non-stationary
    processes, use wavelet-based Hurst estimation as a cross-validation", and
    "Near the Markovian boundary (α ≈ 0.5), cross-validate with wavelet-based
    H estimation".

    BOTH estimates are normalised to H ∈ (0, 1) under the SAME convention
    before differencing. Previously this compared a raw α_DFA (1.50 for a
    random walk) against a convention-converted wavelet H (0.50) and reported
    ``agrees: False`` — a guaranteed spurious disagreement on any
    fBm-convention series, in the very safeguard the spec relies on for the
    Severity-A long-range-correlation falsifier. The two estimators were in
    fact in exact agreement; only the units differed.

    Args:
        signal: Input time series.
        agreement_tolerance: Maximum |H_dfa − H_wavelet| still counted as
            agreement (default 0.15).
        convention: ``"fgn"``, ``"fbm"``, or ``"auto"`` to infer from the DFA
            exponent's magnitude. The SAME convention is applied to both
            estimators, which is what makes the comparison meaningful.

    Returns:
        Dict with ``alpha_dfa`` (the raw exponent, unconverted),
        ``convention`` (the one actually used), ``h_dfa``, ``h_wavelet``,
        ``difference``, ``agrees`` and ``agreement_tolerance``.
    """
    alpha_dfa = estimate_alpha_dfa(signal)
    conv: Convention = infer_convention(alpha_dfa) if convention == "auto" else convention
    h_dfa = hurst_from_alpha_dfa(alpha_dfa, conv)
    # The wavelet path converts a spectral-style slope, which already returns
    # H in (0,1) under its own regime detection; force it onto `conv` so both
    # sides speak the same convention.
    h_wavelet = estimate_hurst_wavelet(signal, convention=conv)
    diff = abs(h_dfa - h_wavelet)
    return {
        "alpha_dfa": alpha_dfa,
        "convention": conv,
        "h_dfa": h_dfa,
        "h_wavelet": h_wavelet,
        "difference": diff,
        "agrees": diff <= agreement_tolerance,
        "agreement_tolerance": agreement_tolerance,
    }


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
