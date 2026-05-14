import numpy as np
import pytest

from stats.spectral_extraction import (
    SpectralSignature,
    bootstrap_confidence_interval,
    compute_aic_bic,
    estimate_hurst_dfa,
    estimate_spectral_exponent_periodogram,
    estimate_spectral_exponent_welch,
    extract_1f_signature,
    print_spectral_signature,
    robust_log_regression,
    validate_hierarchical_spectral_signature,
)


def test_robust_log_regression():
    # Linear data
    x = np.linspace(1, 10, 20)
    y = 2 * x + 1 + 0.1 * np.random.randn(20)
    slope, intercept, r2 = robust_log_regression(x, y)
    assert pytest.approx(slope, rel=0.1) == 2.0

    # With outliers
    y[0] = 100
    slope_out, intercept_out, r2_out = robust_log_regression(x, y)
    assert pytest.approx(slope_out, rel=0.1) == 2.0

    # Error cases
    assert np.isnan(robust_log_regression([1, 2], [1, 2])[0])  # len < 3
    assert np.isnan(robust_log_regression([1, 2, 3], [1, 1, 1])[0])  # std(y) < 1e-10
    assert np.isnan(robust_log_regression([np.nan, 1, 2], [1, 2, 3])[0])


def test_spectral_exponents():
    # Pink noise approximation
    fs = 100.0
    signal = np.cumsum(np.random.randn(500))  # Brown noise (beta ~ 2)

    beta_w, r2_w, h_w = estimate_spectral_exponent_welch(signal, fs=fs)
    assert not np.isnan(beta_w)
    assert beta_w > 0

    beta_p, r2_p, h_p = estimate_spectral_exponent_periodogram(signal, fs=fs)
    assert not np.isnan(beta_p)

    # Short signal error
    assert np.isnan(estimate_spectral_exponent_welch(np.random.randn(10), fmin=40, fmax=50)[0])
    assert np.isnan(
        estimate_spectral_exponent_periodogram(np.random.randn(10), fmin=40, fmax=50)[0]
    )


def test_estimate_hurst_dfa():
    signal = np.cumsum(np.random.randn(400))
    h, r2 = estimate_hurst_dfa(signal)
    assert not np.isnan(h)
    assert h > 0

    # Trigger segments < 2
    h_err, r2_err = estimate_hurst_dfa(np.random.randn(20), min_lag=15, max_lag=16)
    assert np.isnan(h_err)


def test_bootstrap_confidence_interval():
    signal = np.random.randn(100)

    def estimator(x):
        return np.mean(x)

    lower, upper = bootstrap_confidence_interval(signal, estimator, n_bootstrap=50)
    assert lower <= upper

    # Error case
    def fail_estimator(x):
        raise Exception("fail")

    l_err, u_err = bootstrap_confidence_interval(signal, fail_estimator, n_bootstrap=10)
    assert np.isnan(l_err)


def test_compute_aic_bic():
    aic, bic = compute_aic_bic(100, 2, 10.0)
    assert isinstance(aic, float)
    assert isinstance(bic, float)
    assert np.isnan(compute_aic_bic(0, 2, 10.0)[0])
    assert np.isnan(compute_aic_bic(100, 2, 0.0)[0])


def test_extract_1f_signature():
    signal = np.cumsum(np.random.randn(300))
    sig = extract_1f_signature(signal, n_bootstrap=10)
    assert isinstance(sig, SpectralSignature)
    assert sig.beta > 0

    # Custom methods and f range
    sig_custom = extract_1f_signature(
        signal, methods=["welch"], fmin=0.1, fmax=0.4, compute_ci=False
    )
    assert sig_custom.method == "welch"

    # Failed methods
    with pytest.raises(ValueError, match="All spectral estimation methods failed"):
        extract_1f_signature(np.random.randn(5), methods=["welch"])


def test_validate_hierarchical_spectral_signature():
    levels = [np.random.randn(200), np.random.randn(200)]
    res = validate_hierarchical_spectral_signature(levels, fmin=0.1, fmax=10.0)
    assert res["n_levels"] == 2
    assert len(res["signatures"]) == 2
    assert res["coherence_matrix"].shape == (2, 2)

    # Level failure
    res_err = validate_hierarchical_spectral_signature([np.zeros(5)], fmin=0.1, fmax=10.0)
    assert res_err["signatures"][0] is None


def test_print_spectral_signature(capsys):
    sig = SpectralSignature(
        beta=1.0,
        hurst=1.0,
        beta_ci_lower=0.9,
        beta_ci_upper=1.1,
        r_squared=0.9,
        aic=10.0,
        bic=12.0,
        is_pink_noise=True,
        confidence=0.8,
        method="welch",
        n_samples=100,
        frequency_range=(0.1, 10.0),
    )
    print_spectral_signature(sig)
    captured = capsys.readouterr()
    assert "SPECTRAL SIGNATURE EXTRACTION RESULTS" in captured.out


def test_spectral_exceptions():
    # 1. robust_log_regression scaling failure (85)
    # x has no variance -> std < 1e-10
    assert np.isnan(robust_log_regression([1, 1, 1], [1, 2, 3])[0])

    # 2. robust_log_regression LinAlgError (129-130)
    from unittest.mock import patch

    with patch("numpy.polyfit", side_effect=np.linalg.LinAlgError):
        assert np.isnan(robust_log_regression([1, 2, 3, 4], [1, 2, 3, 4])[0])

    # 3. estimate_hurst_dfa LinAlgError (276-277)
    with patch("numpy.polyfit", side_effect=np.linalg.LinAlgError):
        h, r2 = estimate_hurst_dfa(np.cumsum(np.random.randn(100)))
        assert np.isnan(h)

    # 4. extract_1f_signature method failures (426-427, 436-437, 446-447)
    with patch("stats.spectral_extraction.estimate_spectral_exponent_welch", side_effect=Exception):
        sig = extract_1f_signature(
            np.random.randn(200), methods=["welch", "periodogram"], compute_ci=False
        )
        assert sig.method == "periodogram"

    with patch(
        "stats.spectral_extraction.estimate_spectral_exponent_periodogram", side_effect=Exception
    ):
        sig = extract_1f_signature(
            np.random.randn(200), methods=["periodogram", "dfa"], compute_ci=False
        )
        assert sig.method == "dfa"

    with patch("stats.spectral_extraction.estimate_hurst_dfa", side_effect=Exception):
        sig = extract_1f_signature(np.random.randn(200), methods=["dfa", "welch"], compute_ci=False)
        assert sig.method == "welch"

    # 5. beta_estimator exception (467-469) and fallback CI (479-481)
    with patch("stats.spectral_extraction.estimate_spectral_exponent_welch", side_effect=Exception):
        sig = extract_1f_signature(np.random.randn(200), methods=["periodogram"], n_bootstrap=10)
        assert not np.isnan(sig.beta_ci_lower)

    # 6. validate_hierarchical_spectral_signature coherence failure (574-575)
    with patch("scipy.signal.coherence", side_effect=Exception):
        res = validate_hierarchical_spectral_signature([np.random.randn(100), np.random.randn(100)])
        assert res["coherence_matrix"][0, 1] == 0.0
