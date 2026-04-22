"""Comprehensive unit tests for core/preprocessing.py module.

Tests cover:
- RunningStats class
- compute_prediction_error function
- normalize_error function
- z_score function
"""

from __future__ import annotations

import numpy as np
import pytest

from core.preprocessing import (
    RunningStats,
    compute_prediction_error,
    normalize_error,
    z_score,
)


class TestRunningStats:
    """Tests for RunningStats class."""

    def test_initialization(self):
        """Should initialize with correct window size."""
        stats = RunningStats(window_size=10)
        assert len(stats.window) == 0

    def test_invalid_window_size(self):
        """Should raise ValueError for invalid window size."""
        with pytest.raises(ValueError, match="window_size must be > 0"):
            RunningStats(window_size=0)

        with pytest.raises(ValueError, match="window_size must be > 0"):
            RunningStats(window_size=-1)

    def test_update(self):
        """Should add values to window."""
        stats = RunningStats(window_size=3)
        stats.update(1.0)
        stats.update(2.0)
        stats.update(3.0)

        assert len(stats.window) == 3
        assert list(stats.window) == [1.0, 2.0, 3.0]

    def test_window_overflow(self):
        """Should remove oldest values when window is full."""
        stats = RunningStats(window_size=3)
        stats.update(1.0)
        stats.update(2.0)
        stats.update(3.0)
        stats.update(4.0)

        assert len(stats.window) == 3
        assert list(stats.window) == [2.0, 3.0, 4.0]

    def test_mean(self):
        """Should compute mean correctly."""
        stats = RunningStats(window_size=5)
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        for v in values:
            stats.update(v)

        assert stats.mean() == 3.0

    def test_mean_empty_window(self):
        """Should return 0 for empty window."""
        stats = RunningStats(window_size=5)
        assert stats.mean() == 0.0

    def test_variance_bessel_correction(self):
        """Should compute variance with Bessel correction."""
        stats = RunningStats(window_size=5)
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        for v in values:
            stats.update(v)

        # Sample variance with Bessel correction
        expected = np.var(values, ddof=1)
        assert (
            pytest.approx(stats.variance(bessel_correction=True), rel=1e-7) == expected
        )

    def test_variance_mle(self):
        """Should compute variance without Bessel correction."""
        stats = RunningStats(window_size=5)
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        for v in values:
            stats.update(v)

        # Population variance (MLE)
        expected = np.var(values, ddof=0)
        assert (
            pytest.approx(stats.variance(bessel_correction=False), rel=1e-7) == expected
        )

    def test_variance_empty_window(self):
        """Should return 1.0 for empty window."""
        stats = RunningStats(window_size=5)
        assert stats.variance() == 1.0

    def test_std(self):
        """Should compute standard deviation correctly."""
        stats = RunningStats(window_size=5)
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        for v in values:
            stats.update(v)

        expected = np.std(values, ddof=1)
        assert pytest.approx(stats.std(), rel=1e-7) == expected

    def test_running_window_stats(self):
        """Should maintain correct stats with sliding window."""
        stats = RunningStats(window_size=3)

        # Add values one by one and check mean
        stats.update(1.0)
        assert stats.mean() == 1.0

        stats.update(2.0)
        assert stats.mean() == 1.5

        stats.update(3.0)
        assert stats.mean() == 2.0

        stats.update(4.0)  # 1.0 is removed
        assert stats.mean() == 3.0  # mean of [2, 3, 4]


class TestComputePredictionError:
    """Tests for compute_prediction_error function."""

    def test_positive_error(self):
        """Should return positive when x > x_hat."""
        result = compute_prediction_error(x=2.0, x_hat=1.0)
        assert result == 1.0

    def test_negative_error(self):
        """Should return negative when x < x_hat."""
        result = compute_prediction_error(x=1.0, x_hat=2.0)
        assert result == -1.0

    def test_zero_error(self):
        """Should return zero when x equals x_hat."""
        result = compute_prediction_error(x=1.0, x_hat=1.0)
        assert result == 0.0


class TestNormalizeError:
    """Tests for normalize_error function."""

    def test_basic_normalization(self):
        """Should normalize error by sigma."""
        result = normalize_error(z=2.0, sigma=1.0)
        assert pytest.approx(result, rel=1e-5) == 2.0

    def test_small_sigma(self):
        """Should handle small sigma with epsilon."""
        result = normalize_error(z=1.0, sigma=1e-10, eps=1e-8)
        # Uses eps when sigma is very small
        assert result > 0

    def test_negative_sigma(self):
        """Should use abs(sigma)."""
        result = normalize_error(z=2.0, sigma=-1.0)
        assert pytest.approx(result, rel=1e-5) == 2.0

    def test_zero_error(self):
        """Should return zero for zero error."""
        result = normalize_error(z=0.0, sigma=1.0)
        assert result == 0.0


class TestZScore:
    """Tests for z_score function."""

    def test_basic_zscore(self):
        """Should compute z-score correctly."""
        stats = RunningStats(window_size=5)
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            stats.update(v)

        result = z_score(epsilon=6.0, stats=stats)
        # (6.0 - 3.0) / std([1,2,3,4,5])
        expected_std = stats.std()
        expected = (6.0 - 3.0) / expected_std
        assert pytest.approx(result, rel=1e-7) == expected

    def test_zero_at_mean(self):
        """Should return zero when epsilon equals mean."""
        stats = RunningStats(window_size=5)
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            stats.update(v)

        result = z_score(epsilon=3.0, stats=stats)
        assert result == 0.0

    def test_empty_stats(self):
        """Should handle empty stats window."""
        stats = RunningStats(window_size=5)
        result = z_score(epsilon=1.0, stats=stats)
        # With empty window, mean=0, std=1 (default)
        assert result == 1.0

    def test_uses_epsilon(self):
        """Should add eps to std for numerical stability."""
        stats = RunningStats(window_size=5)
        # Add multiple identical values to get zero std
        for _ in range(3):
            stats.update(1.0)

        result = z_score(epsilon=2.0, stats=stats, eps=0.1)
        # Should not crash due to small std, should use eps
        assert abs(result) > 0
