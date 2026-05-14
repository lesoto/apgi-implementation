import numpy as np
import pytest

from stats.hurst import (
    dfa_analysis,
    estimate_beta_welch,
    estimate_hurst_dfa,
    estimate_hurst_robust,
    estimate_spectral_beta,
    hurst_from_slope,
    power_spectrum,
    welch_periodogram,
)


def test_spectral_beta():
    # P(f) = 1/f^2 -> beta = 2
    f = np.array([1, 2, 4, 8])
    p = 1.0 / (f**2)
    beta = estimate_spectral_beta(f, p)
    assert pytest.approx(beta) == 2.0

    with pytest.raises(ValueError, match="need at least two"):
        estimate_spectral_beta([0, 1], [0, 1])


def test_welch_and_beta_welch():
    # Pink noise: beta ~ 1
    fs = 100.0
    # Approximation of 1/f noise
    signal = np.cumsum(np.random.randn(1000))  # Brown noise, beta ~ 2

    f, psd = welch_periodogram(signal, fs=fs)
    assert len(f) > 0

    beta = estimate_beta_welch(signal, fs=fs)
    assert beta > 0  # Should be positive for Brown noise

    # Custom bands
    beta_band = estimate_beta_welch(signal, fs=fs, fmin=1.0, fmax=10.0)
    assert isinstance(beta_band, float)

    # Error case: too few points
    with pytest.raises(ValueError, match="need at least 2 frequency points"):
        estimate_beta_welch(signal[:10], fs=fs, fmin=40, fmax=50)


def test_hurst_from_slope():
    # H = (beta + 1)/2
    assert hurst_from_slope(1.0) == 1.0
    assert hurst_from_slope(0.0) == 0.5


def test_power_spectrum():
    f = np.array([1.0, 10.0])
    taus = np.array([1.0, 0.1])
    sigmas = np.array([1.0, 1.0])
    ps = power_spectrum(f, taus, sigmas)
    assert len(ps) == 2
    assert ps[0] > ps[1]

    with pytest.raises(ValueError, match="same length"):
        power_spectrum(f, [1.0], [1.0, 2.0])


def test_dfa_analysis():
    # Fractal signal: brownian motion has H=0.5 (for increments) or H=1.5 (for process)
    # Actually DFA alpha for brownian motion is ~1.5
    signal = np.cumsum(np.random.randn(1000))
    alpha, scales, F = dfa_analysis(signal)
    assert 1.0 < alpha < 2.0
    assert len(scales) == len(F)

    # Short signal
    with pytest.raises(ValueError, match="signal too short"):
        dfa_analysis(np.zeros(10))

    # Custom scales
    alpha_c, _, _ = dfa_analysis(signal, scales=[10, 20, 30, 40])
    assert isinstance(alpha_c, float)

    # estimate_hurst_dfa wrapper
    h_dfa = estimate_hurst_dfa(signal)
    assert h_dfa == alpha

    # Trigger "fewer than 2 valid scales"
    with pytest.raises(ValueError, match="fewer than 2 valid scales"):
        dfa_analysis(np.random.randn(20), scales=[100, 200])


def test_estimate_hurst_robust():
    signal = np.cumsum(np.random.randn(500))
    h_welch = estimate_hurst_robust(signal, method="welch")
    h_raw = estimate_hurst_robust(signal, method="raw")
    assert isinstance(h_welch, float)
    assert isinstance(h_raw, float)

    with pytest.raises(ValueError, match="unknown method"):
        estimate_hurst_robust(signal, method="invalid")
