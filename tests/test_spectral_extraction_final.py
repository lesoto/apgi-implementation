"""Final tests for stats/spectral_extraction.py to cover remaining lines.

Covers lines 126-128, 421-422.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from stats.spectral_extraction import SpectralSignature, extract_1f_signature, robust_log_regression


class TestRobustLogRegressionEdgeCases:
    """Test edge cases for robust_log_regression function."""

    def test_robust_regression_exact_linear_data(self):
        """Test with perfectly linear data in log-log space."""
        x = np.logspace(-2, 2, 50)
        y = 2.0 * x ** (-1.0)  # Perfect power law
        log_x = np.log(x)
        log_y = np.log(y)

        slope, intercept, r2 = robust_log_regression(log_x, log_y)
        assert not np.isnan(slope)
        assert not np.isnan(r2)
        assert r2 > 0.99  # Should be nearly perfect fit

    def test_robust_regression_with_outliers(self):
        """Test with outlier points that should be downweighted."""
        x = np.logspace(-2, 2, 50)
        y = 2.0 * x ** (-1.0)
        # Add some outliers
        y[10] *= 10.0
        y[20] *= 5.0

        log_x = np.log(x)
        log_y = np.log(y)

        slope, intercept, r2 = robust_log_regression(log_x, log_y)
        assert not np.isnan(slope)
        # Slope should still be close to -1 despite outliers
        assert abs(slope + 1.0) < 0.5


class TestExtract1fSignatureLines126to128:
    """Tests to cover lines 126-128 in spectral_extraction.py."""

    def test_extract_1f_with_confidence_intervals_computed(self):
        """Test extraction where confidence intervals are actually computed."""
        # Pink noise signal
        np.random.seed(42)
        signal = np.cumsum(np.random.randn(1024))  # Brown-like noise

        result = extract_1f_signature(signal, fs=1.0, compute_ci=True)
        assert isinstance(result, SpectralSignature)
        assert not np.isnan(result.beta)

    def test_extract_1f_methods_variations(self):
        """Test with different method combinations."""
        np.random.seed(42)
        signal = np.random.randn(512)

        # Test with single method
        result_welch = extract_1f_signature(signal, fs=1.0, methods=["welch"])
        assert isinstance(result_welch, SpectralSignature)

        result_periodogram = extract_1f_signature(signal, fs=1.0, methods=["periodogram"])
        assert isinstance(result_periodogram, SpectralSignature)


class TestExtract1fSignatureLines421to422:
    """Tests to cover lines 421-422 in spectral_extraction.py."""

    def test_extract_1f_nan_handling_in_regression(self):
        """Test handling of NaN values during regression."""
        np.random.seed(42)
        signal = np.random.randn(256)

        # This should trigger NaN handling paths
        result = extract_1f_signature(signal, fs=1.0)
        assert isinstance(result, SpectralSignature)

    def test_extract_1f_short_signal_edge_case(self):
        """Test with very short signal."""
        np.random.seed(42)
        signal = np.random.randn(32)

        result = extract_1f_signature(signal, fs=1.0)
        # May return NaN for very short signals
        assert isinstance(result, SpectralSignature)

    def test_extract_1f_constant_signal_fallback(self):
        """Test with constant signal (edge case)."""
        signal = np.ones(128)

        # Should handle constant signal gracefully
        with pytest.raises((ValueError, RuntimeError)):
            extract_1f_signature(signal, fs=1.0)


class TestSpectralSignatureDataclass:
    """Test SpectralSignature dataclass."""

    def test_signature_creation(self):
        """Test creating a SpectralSignature."""
        sig = SpectralSignature(
            beta=1.0,
            hurst=1.0,
            beta_ci_lower=0.8,
            beta_ci_upper=1.2,
            r_squared=0.95,
            aic=100.0,
            bic=110.0,
            is_pink_noise=True,
            confidence=0.9,
            method="welch",
            n_samples=1024,
            frequency_range=(0.01, 1.0),
        )
        assert sig.beta == 1.0
        assert sig.is_pink_noise is True

    def test_signature_immutable(self):
        """Test that dataclass is frozen/immutable."""
        sig = SpectralSignature(
            beta=1.0,
            hurst=1.0,
            beta_ci_lower=0.8,
            beta_ci_upper=1.2,
            r_squared=0.95,
            aic=100.0,
            bic=110.0,
            is_pink_noise=True,
            confidence=0.9,
            method="welch",
            n_samples=1024,
            frequency_range=(0.01, 1.0),
        )
        # Should not be able to modify
        with pytest.raises((AttributeError, TypeError, FrozenInstanceError)):
            sig.beta = 2.0


class TestBootstrapConfidenceIntervals:
    """Test bootstrap confidence interval computation."""

    def test_bootstrap_with_sufficient_data(self):
        """Test bootstrap with enough data points."""
        np.random.seed(42)
        signal = np.random.randn(512)

        result = extract_1f_signature(signal, fs=1.0, compute_ci=True, n_bootstrap=50)
        assert isinstance(result, SpectralSignature)
        assert not np.isnan(result.beta_ci_lower)
        assert not np.isnan(result.beta_ci_upper)

    def test_bootstrap_with_small_signal(self):
        """Test bootstrap with small signal (fallback path)."""
        np.random.seed(42)
        signal = np.random.randn(64)

        result = extract_1f_signature(signal, fs=1.0, compute_ci=True, n_bootstrap=10)
        assert isinstance(result, SpectralSignature)
