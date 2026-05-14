from unittest.mock import patch

import numpy as np
import pytest

from stats.spectral_model import (
    SpectralValidator,
    analytic_multiscale_psd,
    compute_psd_1f_exponent_analytic,
    estimate_1f_exponent,
    fit_lorentzian_superposition,
    generate_predicted_spectrum_from_hierarchy,
    hierarchical_spectral_superposition,
    lorentzian_spectrum,
    validate_hurst_dfa,
    validate_pink_noise,
)


def test_lorentzian_spectrum():
    f = np.array([0.1, 1.0, 10.0])
    psd = lorentzian_spectrum(f, tau=1.0, sigma2=1.0)
    assert len(psd) == 3
    assert psd[0] > psd[1] > psd[2]

    with pytest.raises(ValueError, match="tau must be > 0"):
        lorentzian_spectrum(f, tau=0, sigma2=1.0)


def test_analytic_multiscale_psd():
    f = np.logspace(-1, 1, 10)
    taus = np.array([1.0, 0.1])
    sigmas = np.array([1.0, 1.0])
    psd = analytic_multiscale_psd(f, taus, sigmas)
    assert len(psd) == 10

    # Error cases
    with pytest.raises(ValueError, match="same length"):
        analytic_multiscale_psd(f, [1.0], [1.0, 1.0])
    with pytest.raises(ValueError, match="At least one level"):
        analytic_multiscale_psd(f, [], [])
    with pytest.raises(ValueError, match="must be positive"):
        analytic_multiscale_psd(f, [0.0], [1.0])


def test_compute_psd_1f_exponent_analytic():
    taus = np.logspace(-1, 1, 5)
    sigmas = np.ones(5)
    res = compute_psd_1f_exponent_analytic(taus, sigmas)
    assert "beta" in res
    assert res["beta"] > 0


def test_hierarchical_superposition():
    f = np.array([1.0, 2.0])
    taus = np.array([1.0, 0.5])
    sigmas = np.array([1.0, 0.8])
    psd = hierarchical_spectral_superposition(f, taus, sigmas)
    assert len(psd) == 2

    with pytest.raises(ValueError, match="same length"):
        hierarchical_spectral_superposition(f, [1], [1, 2])


def test_estimate_1f_exponent():
    f = np.array([1.0, 2.0, 4.0])
    # P(f) = 1/f^1 -> beta = 1
    psd = 1.0 / f
    beta = estimate_1f_exponent(f, psd)
    assert pytest.approx(beta) == 1.0

    # Range filtering
    beta_range = estimate_1f_exponent(f, psd, fmin=1.5, fmax=5.0)
    assert not np.isnan(beta_range)

    # Error cases
    assert np.isnan(estimate_1f_exponent([1], [1]))  # Too few points


def test_validate_pink_noise():
    f = np.logspace(-1, 1, 10)
    psd = 1.0 / f
    res = validate_pink_noise(f, psd)
    assert res["is_pink_noise"] is True

    # Failed fit case
    res_fail = validate_pink_noise([1], [1])
    assert np.isnan(res_fail["beta"])
    assert res_fail["is_pink_noise"] is False


def test_fit_lorentzian_superposition():
    f = np.logspace(-1, 1, 50)
    taus = np.array([1.0, 0.1])
    true_amps = np.array([0.5, 0.2])
    power = 0.5 * 1.0**2 / (1 + (2 * np.pi * f * 1.0) ** 2) + 0.2 * 0.1**2 / (
        1 + (2 * np.pi * f * 0.1) ** 2
    )

    res = fit_lorentzian_superposition(f, power, taus)
    assert pytest.approx(res["amplitudes"], rel=0.1) == true_amps.tolist()
    assert res["r_squared"] > 0.9

    # Fit failure (RuntimeError in curve_fit)
    with patch("scipy.optimize.curve_fit", side_effect=RuntimeError):
        res_fail = fit_lorentzian_superposition(f, power, taus)
        assert res_fail["r_squared"] < 1.0  # Should have returned initial guess


def test_generate_predicted_spectrum():
    psd, taus, sigmas = generate_predicted_spectrum_from_hierarchy(np.array([1.0, 10.0]))
    assert len(psd) == 2
    assert len(taus) == 5


def test_validate_hurst_dfa():
    signal = np.cumsum(np.random.randn(200))
    res = validate_hurst_dfa(signal)
    assert "hurst" in res

    # DFA failure
    res_fail = validate_hurst_dfa(np.zeros(10))
    assert np.isnan(res_fail["hurst"])


def test_spectral_validator():
    val = SpectralValidator(n_levels=3)
    signal = np.cumsum(np.random.randn(200))
    res = val.validate_signal(signal)
    assert "beta_observed" in res

    # Periodogram method
    res_p = val.validate_signal(signal, method="periodogram")
    assert "beta_observed" in res_p

    # Plotting
    fig = val.plot_comparison(signal)
    assert fig is not None

    # Simplified mock for plot_comparison failure
    with patch("matplotlib.pyplot.subplots", side_effect=ImportError):
        assert val.plot_comparison(signal) is None


def test_spectral_model_exceptions():
    # estimate_1f_exponent LinAlgError (236)
    with patch("numpy.polyfit", side_effect=np.linalg.LinAlgError):
        assert np.isnan(estimate_1f_exponent([1, 2, 3], [1, 2, 3]))
