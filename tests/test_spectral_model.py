"""Tests for stats/spectral_model.py spectral analysis."""

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
    validate_pink_noise,
)


class TestLorentzianSpectrum:
    """Tests for lorentzian_spectrum function."""

    def test_basic_computation(self):
        """Test basic Lorentzian spectrum computation."""
        f = np.array([0.1, 1.0, 10.0])
        tau = 1.0
        sigma2 = 1.0
        result = lorentzian_spectrum(f, tau, sigma2)
        assert len(result) == len(f)
        assert np.all(result > 0)

    def test_tau_validation(self):
        """Test that tau <= 0 raises ValueError."""
        f = np.array([1.0])
        with pytest.raises(ValueError, match="tau must be > 0"):
            lorentzian_spectrum(f, tau=0, sigma2=1.0)
        with pytest.raises(ValueError, match="tau must be > 0"):
            lorentzian_spectrum(f, tau=-1, sigma2=1.0)

    def test_zero_frequency(self):
        """Test Lorentzian at zero frequency."""
        f = np.array([0.0])
        tau = 1.0
        sigma2 = 1.0
        result = lorentzian_spectrum(f, tau, sigma2)
        # At f=0: P(0) = sigma2 * tau^2
        expected = sigma2 * tau**2
        assert result[0] == pytest.approx(expected)

    def test_high_frequency_decay(self):
        """Test that spectrum decays at high frequencies."""
        f = np.array([0.1, 100.0])
        tau = 1.0
        sigma2 = 1.0
        result = lorentzian_spectrum(f, tau, sigma2)
        # High frequency should have lower power
        assert result[1] < result[0]


class TestAnalyticMultiscalePSD:
    """Tests for analytic_multiscale_psd function."""

    def test_basic_computation(self):
        """Test basic multi-timescale PSD computation."""
        f = np.logspace(-2, 2, 100)
        taus = np.array([0.1, 1.0, 10.0])
        sigma2s = np.array([1.0, 1.0, 1.0])
        result = analytic_multiscale_psd(f, taus, sigma2s)
        assert len(result) == len(f)
        assert np.all(result >= 0)

    def test_length_mismatch(self):
        """Test that mismatched lengths raise ValueError."""
        f = np.array([1.0])
        taus = np.array([1.0, 2.0])
        sigma2s = np.array([1.0])
        with pytest.raises(ValueError, match="taus and sigma2s must have same length"):
            analytic_multiscale_psd(f, taus, sigma2s)

    def test_empty_arrays(self):
        """Test that empty arrays raise ValueError."""
        f = np.array([1.0])
        taus = np.array([])
        sigma2s = np.array([])
        with pytest.raises(ValueError, match="At least one level required"):
            analytic_multiscale_psd(f, taus, sigma2s)

    def test_negative_tau(self):
        """Test that negative tau raises ValueError."""
        f = np.array([1.0])
        taus = np.array([-1.0])
        sigma2s = np.array([1.0])
        with pytest.raises(ValueError, match="All time constants must be positive"):
            analytic_multiscale_psd(f, taus, sigma2s)

    def test_zero_tau(self):
        """Test that zero tau raises ValueError."""
        f = np.array([1.0])
        taus = np.array([0.0])
        sigma2s = np.array([1.0])
        with pytest.raises(ValueError, match="All time constants must be positive"):
            analytic_multiscale_psd(f, taus, sigma2s)


class TestComputePSD1FExponentAnalytic:
    """Tests for compute_psd_1f_exponent_analytic function."""

    def test_basic_computation(self):
        """Test basic 1/f exponent computation."""
        taus = np.logspace(-2, 1, 5)
        sigma2s = np.ones(5) * 0.1
        result = compute_psd_1f_exponent_analytic(taus, sigma2s)
        assert "beta" in result
        assert "hurst" in result
        assert "freqs" in result
        assert "psd" in result
        assert "f_range" in result
        assert result["beta"] > 0

    def test_custom_frequency_range(self):
        """Test with custom frequency range."""
        taus = np.array([0.1, 1.0, 10.0])
        sigma2s = np.ones(3)
        result = compute_psd_1f_exponent_analytic(
            taus, sigma2s, f_range=(0.001, 100.0), n_points=500
        )
        assert len(result["freqs"]) == 500


class TestHierarchicalSpectralSuperposition:
    """Tests for hierarchical_spectral_superposition function."""

    def test_basic_computation(self):
        """Test basic spectral superposition."""
        freqs = np.logspace(-2, 2, 100)
        taus = np.array([0.1, 1.0, 10.0])
        sigma2s = np.ones(3)
        result = hierarchical_spectral_superposition(freqs, taus, sigma2s)
        assert len(result) == len(freqs)
        assert np.all(result >= 0)

    def test_length_mismatch(self):
        """Test that mismatched lengths raise ValueError."""
        freqs = np.array([1.0])
        taus = np.array([1.0, 2.0])
        sigma2s = np.array([1.0])
        with pytest.raises(ValueError, match="taus and sigma2s must have same length"):
            hierarchical_spectral_superposition(freqs, taus, sigma2s)


class TestEstimate1FExponent:
    """Tests for estimate_1f_exponent function."""

    def test_basic_computation(self):
        """Test basic 1/f exponent estimation."""
        freqs = np.logspace(-2, 2, 100)
        # Create 1/f^beta noise: P(f) = 1/f^beta
        beta_true = 1.0
        psd = freqs ** (-beta_true)
        beta_est = estimate_1f_exponent(freqs, psd)
        assert 0.5 < beta_est < 1.5  # Should be close to 1.0

    def test_frequency_range_selection(self):
        """Test with frequency range constraints."""
        freqs = np.logspace(-2, 2, 100)
        psd = freqs ** (-1.0)
        beta_est = estimate_1f_exponent(freqs, psd, fmin=0.1, fmax=10.0)
        assert beta_est > 0

    def test_insufficient_points(self):
        """Test that insufficient points return NaN."""
        freqs = np.array([1.0])
        psd = np.array([1.0])
        beta_est = estimate_1f_exponent(freqs, psd)
        assert np.isnan(beta_est)

    def test_all_positive_mask(self):
        """Test with PSD that has all positive values."""
        freqs = np.logspace(-1, 1, 50)
        psd = np.ones(50) * 1.0
        beta_est = estimate_1f_exponent(freqs, psd)
        # Flat spectrum should give beta ≈ 0
        assert abs(beta_est) < 0.5

    def test_linalg_error_handling(self, suppress_lapack):
        """Test handling of np.linalg.LinAlgError."""
        # Create data that will cause LinAlgError in polyfit
        # This happens when the matrix is singular
        with pytest.warns(RuntimeWarning):
            freqs = np.array([1.0, 1.0, 1.0])  # Duplicate frequencies
            psd = np.array([1.0, 1.0, 1.0])
            beta_est = estimate_1f_exponent(freqs, psd)
            # Should return NaN when LinAlgError occurs
            assert np.isnan(beta_est)


class TestValidatePinkNoise:
    """Tests for validate_pink_noise function."""

    def test_valid_pink_noise(self):
        """Test validation of pink noise spectrum."""
        freqs = np.logspace(-2, 2, 100)
        psd = freqs ** (-1.0)  # True 1/f noise
        result = validate_pink_noise(freqs, psd)
        assert "beta" in result
        assert "hurst_exponent" in result
        assert "is_pink_noise" in result
        assert result["is_pink_noise"] is True

    def test_invalid_pink_noise(self):
        """Test validation of non-pink noise spectrum."""
        freqs = np.logspace(-2, 2, 100)
        psd = freqs ** (-2.0)  # 1/f^2 noise (too steep)
        result = validate_pink_noise(freqs, psd, beta_target=1.0, tolerance=0.3)
        assert result["is_pink_noise"] is False

    def test_nan_beta_handling(self):
        """Test handling of NaN beta from fitting failure."""
        freqs = np.array([1.0, 2.0])
        psd = np.array([0.0, 0.0])  # Zero PSD will cause issues
        # With only 2 points and zero PSD, should handle gracefully
        result = validate_pink_noise(freqs, psd)
        assert "beta" in result
        assert "is_pink_noise" in result

    def test_with_frequency_range(self):
        """Test with specific frequency range."""
        freqs = np.logspace(-2, 2, 100)
        psd = freqs ** (-1.0)
        result = validate_pink_noise(freqs, psd, fmin=0.1, fmax=10.0)
        assert result["frequency_range"] == (0.1, 10.0)


class TestFitLorentzianSuperposition:
    """Tests for fit_lorentzian_superposition function."""

    def test_basic_fit(self):
        """Test basic Lorentzian superposition fitting."""
        freqs = np.logspace(-2, 2, 100)
        taus = np.array([0.1, 1.0, 10.0])
        # Generate synthetic data
        sigma2s = np.array([0.5, 1.0, 0.5])
        power = hierarchical_spectral_superposition(freqs, taus, sigma2s)
        # Add small noise
        power += np.random.randn(len(power)) * 0.01

        result = fit_lorentzian_superposition(freqs, power, taus)
        assert "amplitudes" in result
        assert "fitted_psd" in result
        assert "residuals" in result
        assert "r_squared" in result
        assert len(result["amplitudes"]) == len(taus)
        assert 0 <= result["r_squared"] <= 1

    def test_fit_failure_handling(self):
        """Test handling of fit failures."""
        freqs = np.array([1.0])  # Too few points for proper fit
        power = np.array([1.0])
        taus = np.array([1.0])
        # This may fail to converge, but should not crash
        result = fit_lorentzian_superposition(freqs, power, taus)
        assert "amplitudes" in result
        assert "r_squared" in result

    def test_runtime_error_handling(self):
        """Test RuntimeError handling when curve_fit fails."""
        # Create data that will cause RuntimeError in curve_fit
        freqs = np.array([1.0, 2.0, 3.0])
        power = np.array([1e10, 1e10, 1e10])  # Very large values
        taus = np.array([1.0, 2.0])
        # Should handle RuntimeError and return initial guess
        result = fit_lorentzian_superposition(freqs, power, taus)
        assert "amplitudes" in result
        assert len(result["amplitudes"]) == len(taus)


class TestGeneratePredictedSpectrumFromHierarchy:
    """Tests for generate_predicted_spectrum_from_hierarchy function."""

    def test_basic_generation(self):
        """Test basic spectrum generation."""
        freqs = np.logspace(-2, 2, 100)
        psd, taus, sigma2s = generate_predicted_spectrum_from_hierarchy(
            freqs, n_levels=5, tau_min=0.01, tau_max=10.0
        )
        assert len(psd) == len(freqs)
        assert len(taus) == 5
        assert len(sigma2s) == 5
        assert np.all(psd >= 0)

    def test_sigma2_range(self):
        """Test with custom sigma2 range."""
        freqs = np.logspace(-2, 2, 50)
        psd, taus, sigma2s = generate_predicted_spectrum_from_hierarchy(
            freqs, n_levels=3, sigma2_range=(0.5, 2.0)
        )
        assert len(sigma2s) == 3
        assert sigma2s[0] > sigma2s[-1]  # Decreasing with timescale


class TestSpectralValidator:
    """Tests for SpectralValidator class."""

    def test_initialization(self):
        """Test SpectralValidator initialization."""
        validator = SpectralValidator(n_levels=5, tau_min=0.01, tau_max=10.0)
        assert validator.n_levels == 5
        assert validator.tau_min == 0.01
        assert validator.tau_max == 10.0
        assert len(validator.taus) == 5
        assert len(validator.freqs) == 1000

    def test_validate_signal(self):
        """Test signal validation."""
        validator = SpectralValidator()
        # Generate synthetic pink noise signal
        signal = np.random.randn(1000)
        # Make it more like 1/f by filtering
        result = validator.validate_signal(signal, fs=100.0)
        assert "beta_observed" in result
        assert "beta_predicted" in result
        assert "beta_error" in result
        assert "matches_prediction" in result

    def test_validate_signal_periodogram(self):
        """Test signal validation with periodogram method."""
        validator = SpectralValidator()
        signal = np.random.randn(1000)
        result = validator.validate_signal(signal, fs=100.0, method="periodogram")
        assert "beta_observed" in result

    def test_plot_comparison(self):
        """Test plot comparison functionality."""
        validator = SpectralValidator()
        signal = np.random.randn(1000)

        import matplotlib.pyplot as plt  # type: ignore[import-untyped]

        fig = validator.plot_comparison(signal, fs=100.0)
        # If matplotlib is available, should return figure
        assert fig is not None
        assert hasattr(fig, "axes")
        plt.close(fig)

    def test_plot_comparison_with_matplotlib_available(self):
        """Test plot comparison when matplotlib is available."""
        validator = SpectralValidator()
        signal = np.random.randn(1000)

        # Force matplotlib to be available
        import matplotlib.pyplot as plt  # type: ignore[import-untyped]

        fig = validator.plot_comparison(signal, fs=100.0)
        # Should return a figure object
        assert fig is not None
        assert hasattr(fig, "axes")
        plt.close(fig)

    def test_plot_comparison_matplotlib_import_error(self):
        """Test plot comparison handles ImportError when matplotlib unavailable."""
        import sys
        from unittest.mock import patch

        validator = SpectralValidator()
        signal = np.random.randn(1000)

        # Mock matplotlib import to raise ImportError
        with patch.dict(sys.modules, {"matplotlib.pyplot": None}):
            result = validator.plot_comparison(signal, fs=100.0)
            # Should return None when matplotlib is not available
            assert result is None
            # Ensure the result is explicitly None
            assert result is None or result is False

    def test_plot_comparison_import_error_handling(self):
        """Test plot comparison handles ImportError gracefully."""
        validator = SpectralValidator()
        signal = np.random.randn(1000)

        # This should not raise even if matplotlib is not available
        fig = validator.plot_comparison(signal, fs=100.0)
        # If matplotlib is available, check figure
        if fig is not None:
            assert hasattr(fig, "axes")

    def test_plot_comparison_no_matplotlib(self):
        """Test plot comparison when matplotlib is not available."""
        import sys
        from unittest.mock import patch

        validator = SpectralValidator()
        signal = np.random.randn(1000)

        # Mock matplotlib import to raise ImportError
        with patch.dict(sys.modules, {"matplotlib.pyplot": None}):
            # This should trigger the ImportError path
            result = validator.plot_comparison(signal, fs=100.0)
            # Should return None when matplotlib is not available
            assert result is None
            # Ensure we explicitly check the result
            assert result is None

    def test_fit_lorentzian_superposition_runtime_error(self):
        """Test fitting handles RuntimeError gracefully."""
        from stats.spectral_model import fit_lorentzian_superposition

        # Create data that might cause fitting to fail
        freqs = np.logspace(-2, 2, 100)
        power = np.abs(np.random.randn(100))  # Use positive data
        taus = np.array([0.1, 1.0, 10.0])

        result = fit_lorentzian_superposition(freqs, power, taus)
        # Should still return a result even if fitting fails
        assert "amplitudes" in result
        assert "fitted_psd" in result
        assert "residuals" in result
        assert "r_squared" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])  # pragma: no cover
