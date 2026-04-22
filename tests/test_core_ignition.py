"""Comprehensive unit tests for core/ignition.py module.

Tests cover:
- compute_ignition_probability function
- sample_ignition_state function
- detect_ignition_event function
- compute_margin function
"""

from __future__ import annotations

import numpy as np
import pytest

from core.ignition import (
    compute_ignition_probability,
    sample_ignition_state,
    detect_ignition_event,
    compute_margin,
)


class TestComputeIgnitionProbability:
    """Tests for compute_ignition_probability function."""

    def test_sigmoid_center(self):
        """Should return 0.5 when S equals theta."""
        result = compute_ignition_probability(1.0, 1.0, tau=0.5)
        assert result == 0.5

    def test_sigmoid_high_signal(self):
        """Should approach 1.0 when S >> theta."""
        result = compute_ignition_probability(10.0, 1.0, tau=0.5)
        assert result > 0.99

    def test_sigmoid_low_signal(self):
        """Should approach 0.0 when S << theta."""
        result = compute_ignition_probability(-2.0, 1.0, tau=0.5)
        assert result < 0.01

    def test_sigmoid_temperature_effect(self):
        """Should show sensitivity to tau parameter."""
        # Lower tau = steeper sigmoid
        result_steep = compute_ignition_probability(1.2, 1.0, tau=0.1)
        result_shallow = compute_ignition_probability(1.2, 1.0, tau=1.0)
        assert result_steep > result_shallow

    def test_invalid_tau(self):
        """Should raise ValueError for non-positive tau."""
        with pytest.raises(ValueError, match="tau must be > 0"):
            compute_ignition_probability(1.0, 0.5, tau=0.0)

        with pytest.raises(ValueError, match="tau must be > 0"):
            compute_ignition_probability(1.0, 0.5, tau=-0.5)

    def test_extreme_values(self):
        """Should handle extreme input values without overflow."""
        # Very large positive difference
        result_large = compute_ignition_probability(1000.0, 0.0, tau=0.5)
        assert result_large <= 1.0

        # Very large negative difference
        result_small = compute_ignition_probability(-1000.0, 0.0, tau=0.5)
        assert result_small >= 0.0


class TestSampleIgnitionState:
    """Tests for sample_ignition_state function."""

    def test_always_ignite(self):
        """Should always return 1 when p_ignite=1.0."""
        for _ in range(10):
            result = sample_ignition_state(1.0)
            assert result == 1

    def test_never_ignite(self):
        """Should always return 0 when p_ignite=0.0."""
        for _ in range(10):
            result = sample_ignition_state(0.0)
            assert result == 0

    def test_probabilistic_ignition(self):
        """Should return 1 with probability p_ignite."""
        np.random.seed(42)
        p_ignite = 0.5
        n_samples = 1000
        results = [sample_ignition_state(p_ignite) for _ in range(n_samples)]
        empirical_rate = sum(results) / n_samples
        # Should be close to 0.5 within tolerance
        assert 0.4 < empirical_rate < 0.6

    def test_invalid_probability(self):
        """Should raise ValueError for invalid probabilities."""
        with pytest.raises(ValueError, match="p_ignite must be in"):
            sample_ignition_state(-0.1)

        with pytest.raises(ValueError, match="p_ignite must be in"):
            sample_ignition_state(1.1)

    def test_with_custom_rng(self):
        """Should use provided RNG when given."""
        rng = np.random.default_rng(42)
        result1 = sample_ignition_state(0.5, rng=rng)

        rng = np.random.default_rng(42)
        result2 = sample_ignition_state(0.5, rng=rng)

        assert result1 == result2


class TestDetectIgnitionEvent:
    """Tests for detect_ignition_event function."""

    def test_ignition_when_superthreshold(self):
        """Should return True when S > theta."""
        assert detect_ignition_event(1.5, 1.0) is True
        assert detect_ignition_event(2.0, 1.0) is True

    def test_no_ignition_when_subthreshold(self):
        """Should return False when S <= theta."""
        assert detect_ignition_event(1.0, 1.0) is False
        assert detect_ignition_event(0.5, 1.0) is False

    def test_boundary_condition(self):
        """Should handle S == theta."""
        assert detect_ignition_event(1.0, 1.0) is False

    def test_negative_values(self):
        """Should handle negative signal and threshold."""
        assert detect_ignition_event(-0.5, -1.0) is True
        assert detect_ignition_event(-1.5, -1.0) is False


class TestComputeMargin:
    """Tests for compute_margin function."""

    def test_positive_margin(self):
        """Should return positive when S > theta."""
        result = compute_margin(1.5, 1.0)
        assert result == 0.5
        assert result > 0

    def test_negative_margin(self):
        """Should return negative when S < theta."""
        result = compute_margin(0.5, 1.0)
        assert result == -0.5
        assert result < 0

    def test_zero_margin(self):
        """Should return zero when S == theta."""
        result = compute_margin(1.0, 1.0)
        assert result == 0.0

    def test_margin_calculation(self):
        """Should compute S - theta correctly."""
        assert compute_margin(2.0, 0.5) == 1.5
        assert compute_margin(0.0, 1.0) == -1.0

    def test_with_extreme_values(self):
        """Should handle extreme values."""
        assert compute_margin(1e10, 0.0) == 1e10
        assert compute_margin(0.0, 1e10) == -1e10
