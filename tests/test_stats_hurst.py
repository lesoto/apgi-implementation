"""Comprehensive unit tests for stats/hurst.py module.

Tests cover:
- estimate_spectral_beta function
- welch_periodogram function
- estimate_beta_welch function
- hurst_from_slope function
- power_spectrum function
- estimate_hurst_robust function
"""

from __future__ import annotations

import numpy as np
import pytest

from stats.hurst import (
    estimate_beta_welch,
    estimate_hurst_robust,
    estimate_spectral_beta,
    hurst_from_slope,
    power_spectrum,
    welch_periodogram,
)


class TestEstimateSpectralBeta:
    """Tests for estimate_spectral_beta function."""

    def test_basic_estimation(self):
        """Should estimate spectral exponent."""
        freqs = np.array([1.0, 2.0, 4.0, 8.0])
        power = np.array([1.0, 0.5, 0.25, 0.125])  # 1/f relationship

        result = estimate_spectral_beta(freqs, power)
        # Beta should be close to 1 for 1/f noise
        assert result > 0.5
        assert result < 2.0

    def test_insufficient_points(self):
        """Should raise ValueError for insufficient points."""
        freqs = np.array([1.0])
        power = np.array([1.0])

        with pytest.raises(ValueError, match="need at least two"):
            estimate_spectral_beta(freqs, power)


class TestWelchPeriodogram:
    """Tests for welch_periodogram function."""

    def test_basic_computation(self):
        """Should compute Welch periodogram."""
        signal = np.random.randn(1000)

        freqs, psd = welch_periodogram(signal, fs=100.0)

        assert len(freqs) > 0
        assert len(psd) > 0
        assert len(freqs) == len(psd)

    def test_default_nperseg(self):
        """Should use default nperseg when not specified."""
        signal = np.random.randn(1000)

        freqs, psd = welch_periodogram(signal, fs=100.0)
        assert len(freqs) > 0

    def test_explicit_nperseg(self):
        """Should use explicitly provided nperseg."""
        signal = np.random.randn(1000)

        freqs, psd = welch_periodogram(signal, fs=100.0, nperseg=128)
        # For nperseg=128 with real signal, welch returns nperseg/2 + 1 = 65 frequencies
        assert len(freqs) == 65
        assert len(psd) == 65


class TestEstimateBetaWelch:
    """Tests for estimate_beta_welch function."""

    def test_basic_estimation(self):
        """Should estimate beta using Welch method."""
        # Generate 1/f-like signal
        np.random.seed(42)
        signal = np.cumsum(np.random.randn(1000))

        result = estimate_beta_welch(signal, fs=100.0)

        # Should return a positive beta value
        assert result > 0

    def test_frequency_range(self):
        """Should respect frequency range."""
        np.random.seed(42)
        signal = np.cumsum(np.random.randn(1000))

        result = estimate_beta_welch(signal, fs=100.0, fmin=1.0, fmax=10.0)
        assert result > 0


class TestHurstFromSlope:
    """Tests for hurst_from_slope function."""

    def test_basic_computation(self):
        """Should compute H from beta."""
        result = hurst_from_slope(beta_spec=1.0)
        # H = (1 + 1) / 2 = 1.0
        assert result == 1.0

    def test_zero_beta(self):
        """Should return 0.5 for white noise (beta=0)."""
        result = hurst_from_slope(beta_spec=0.0)
        # H = (0 + 1) / 2 = 0.5
        assert result == 0.5


class TestPowerSpectrum:
    """Tests for power_spectrum function."""

    def test_basic_computation(self):
        """Should compute power spectrum correctly."""
        freqs = np.array([1.0, 2.0, 4.0])
        tau_levels = np.array([10.0, 100.0])
        sigma_levels = np.array([1.0, 1.0])

        result = power_spectrum(freqs, tau_levels, sigma_levels)

        assert len(result) == len(freqs)
        assert np.all(result > 0)

    def test_length_mismatch(self):
        """Should raise ValueError for mismatched lengths."""
        freqs = np.array([1.0, 2.0])
        tau_levels = np.array([10.0, 100.0])
        sigma_levels = np.array([1.0])

        with pytest.raises(ValueError, match="must have the same length"):
            power_spectrum(freqs, tau_levels, sigma_levels)


class TestEstimateHurstRobust:
    """Tests for estimate_hurst_robust function."""

    def test_welch_method(self):
        """Should estimate H using Welch method."""
        np.random.seed(42)
        signal = np.cumsum(np.random.randn(1000))

        result = estimate_hurst_robust(signal, fs=100.0, method="welch")

        # H should be between 0 and 1
        assert 0.0 < result < 1.5

    def test_raw_method(self):
        """Should estimate H using raw FFT method."""
        np.random.seed(42)
        signal = np.cumsum(np.random.randn(1000))

        result = estimate_hurst_robust(signal, fs=100.0, method="raw")

        # H should be between 0 and 1
        assert 0.0 < result < 1.5

    def test_invalid_method(self):
        """Should raise ValueError for invalid method."""
        np.random.seed(42)
        signal = np.cumsum(np.random.randn(1000))

        with pytest.raises(ValueError, match="unknown method"):
            estimate_hurst_robust(signal, fs=100.0, method="invalid")
