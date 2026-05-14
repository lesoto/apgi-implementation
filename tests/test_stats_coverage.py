import pytest
import numpy as np
from stats.maturity_assessment import (
    assess_hierarchical_architecture,
    assess_statistical_validation,
    assess_overall_maturity,
    log_maturity_assessment,
    format_maturity_assessment,
    get_maturity_rating,
    MaturityScore,
)
from stats.spectral_extraction import (
    extract_1f_signature,
    validate_hierarchical_spectral_signature,
    print_spectral_signature,
    robust_log_regression,
)


def test_assess_hierarchical_architecture():
    # Test with multiple levels
    signal_levels = [np.random.randn(200) for _ in range(3)]
    theta_levels = [np.random.randn(200) for _ in range(3)]
    phi_levels = [np.random.randn(200) for _ in range(3)]
    pi_levels = [np.random.randn(200) for _ in range(3)]

    pac, cascade, coh, issues, recs = assess_hierarchical_architecture(
        signal_levels, theta_levels, phi_levels, pi_levels
    )
    assert pac >= 0
    assert cascade >= 0
    assert coh >= 0

    # Test with single level
    pac2, cascade2, coh2, issues2, recs2 = assess_hierarchical_architecture(
        [signal_levels[0]], [theta_levels[0]], [phi_levels[0]], [pi_levels[0]]
    )
    assert pac2 == 0
    assert cascade2 == 0
    assert coh2 == 0


def test_assess_statistical_validation():
    sig = np.random.randn(200)
    spec, conf, cons, issues, recs, signature = assess_statistical_validation(sig)
    assert spec >= 0
    assert conf >= 0

    # Test with signal levels
    signal_levels = [np.random.randn(200) for _ in range(2)]
    spec3, conf3, cons3, issues3, recs3, signature3 = assess_statistical_validation(
        sig, signal_levels
    )
    assert cons3 >= 0

    # Test short signal
    spec2, conf2, cons2, issues2, recs2, sig2 = assess_statistical_validation(np.zeros(50))
    assert spec2 == 0
    assert "too short" in issues2[0]


def test_assess_overall_maturity():
    sig = np.random.randn(200)
    score = assess_overall_maturity(sig)
    assert isinstance(score, MaturityScore)
    assert score.overall_score >= 0

    # Test with low score for recs
    score_low = assess_overall_maturity(np.zeros(200))
    assert any("maturity is low" in r for r in score_low.recommendations)

    # Test formatting
    fmt = format_maturity_assessment(score)
    assert "MATURITY ASSESSMENT" in fmt

    # Test logging
    log_maturity_assessment(score)

    # Test rating
    rating = get_maturity_rating(score.overall_score)
    assert isinstance(rating, str)


def test_maturity_rating_thresholds():
    assert "Excellent" in get_maturity_rating(95)
    assert "Very Good" in get_maturity_rating(85)
    assert "Good" in get_maturity_rating(75)
    assert "Satisfactory" in get_maturity_rating(65)
    assert "Acceptable" in get_maturity_rating(55)
    assert "Needs Improvement" in get_maturity_rating(45)


def test_spectral_extraction_robustness():
    # Test spectral extraction with non-pink noise
    sig = np.random.randn(200)
    sig_ext = extract_1f_signature(sig)
    assert hasattr(sig_ext, "beta")

    # Test without bootstrap
    sig_ext2 = extract_1f_signature(sig, compute_ci=False)
    assert sig_ext2.beta_ci_lower != 0

    # Test with constant signal - should raise ValueError
    sig_const = np.ones(200)
    with pytest.raises(ValueError, match="All spectral estimation methods failed"):
        extract_1f_signature(sig_const)


def test_hierarchical_spectral_validation():
    signal_levels = [np.random.randn(200) for _ in range(2)]
    res = validate_hierarchical_spectral_signature(signal_levels, fmin=0.1, fmax=10.0)
    assert "signatures" in res
    assert "coherence_matrix" in res

    # Test with failed extraction
    res2 = validate_hierarchical_spectral_signature([np.ones(10)], fs=1.0)
    assert res2["signatures"][0] is None


def test_print_spectral_signature():
    sig = extract_1f_signature(np.random.randn(200))
    print_spectral_signature(sig)


def test_robust_regression_edge_cases():
    # Insufficient data
    s, i, r = robust_log_regression(np.array([1, 2]), np.array([1, 2]))
    assert np.isnan(s)

    # Constant x or y
    s2, i2, r2 = robust_log_regression(np.array([1, 1, 1]), np.array([1, 2, 3]))
    assert np.isnan(s2)

    # Outliers
    x = np.linspace(0, 10, 20)
    y = 2 * x + 1
    y[5] += 100  # Huge outlier
    s3, i3, r3 = robust_log_regression(x, y)
    assert abs(s3 - 2.0) < 0.5  # Should be robust


def test_stats_missing_branches():
    # Test pink noise (beta near 1.0)
    fs = 100.0
    T = 2.0
    t = np.arange(0, T, 1 / fs)
    # Simple pink noise approximation: 1/f in frequency domain
    white = np.random.randn(len(t))
    f = np.fft.rfftfreq(len(t), 1 / fs)
    f[0] = f[1]  # avoid div by zero
    pink_filt = 1 / np.sqrt(f)
    pink_fft = np.fft.rfft(white) * pink_filt
    pink = np.fft.irfft(pink_fft)

    score = assess_overall_maturity(pink, fs=fs)
    # R2 might still be low for this simple approx, but let's check it doesn't crash
    assert score.spectral_signature is not None

    # Test high beta (> 1.5)
    brown = np.cumsum(np.random.randn(200))
    score_brown = assess_overall_maturity(brown)
    assert any("too high" in i for i in score_brown.issues)

    # Test low beta (< 0.8)
    score_white = assess_overall_maturity(np.random.randn(200))
    assert any("too low" in i for i in score_white.issues)

    # Test empty correlations for PAC and cascade
    res = assess_hierarchical_architecture(
        [np.array([1, 1]), np.array([1, 1])],
        [np.array([1, 1]), np.array([1, 1])],
        [np.array([1, 1]), np.array([1, 1])],
        [np.array([1, 1]), np.array([1, 1])],
    )
    assert res[0] == 0  # pac_score
    assert res[1] == 0  # cascade_score


def test_stats_exception_branches(mocker):
    # Mock extract_1f_signature to fail
    mocker.patch(
        "stats.maturity_assessment.extract_1f_signature", side_effect=Exception("Mock failure")
    )

    sig = np.random.randn(200)
    # This should trigger the exception handler in assess_statistical_validation
    spec, conf, cons, issues, recs, signature = assess_statistical_validation(sig)
    assert spec == 0
    assert any("Spectral extraction failed" in i for i in issues)

    # Trigger consistency score exception
    spec2, conf2, cons2, issues2, recs2, signature2 = assess_statistical_validation(sig, [sig, sig])
    assert cons2 == 0


def test_spectral_extraction_more_failures(mocker):
    from stats.spectral_extraction import compute_aic_bic

    # AIC/BIC div by zero
    a, b = compute_aic_bic(0, 2, 0)
    assert np.isnan(a)

    # Welch failure in extract_1f_signature
    mocker.patch(
        "stats.spectral_extraction.estimate_spectral_exponent_welch",
        side_effect=Exception("Welch failed"),
    )
    # Periodogram and DFA still work
    sig = np.random.randn(200)
    res = extract_1f_signature(sig)
    assert res.method in ["periodogram", "dfa"]

    # Bootstrap failure triggers fallback
    mocker.patch(
        "stats.spectral_extraction.bootstrap_confidence_interval", return_value=(np.nan, np.nan)
    )
    res2 = extract_1f_signature(sig)
    assert not np.isnan(res2.beta_ci_lower)  # Should use fallback
