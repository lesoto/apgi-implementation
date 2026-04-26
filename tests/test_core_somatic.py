"""Comprehensive unit tests for core/somatic.py module.

Tests cover:
- somatic_marker_arousal function
- somatic_marker_valence function
- compute_precision_with_somatic_marker function
- compute_somatic_gain function
- update_somatic_marker_euler function
"""

from __future__ import annotations

import numpy as np
import pytest

from core.somatic import (
    compute_precision_with_somatic_marker,
    compute_somatic_gain,
    somatic_marker_arousal,
    somatic_marker_valence,
    update_somatic_marker_euler,
)


class TestSomaticMarkerArousal:
    """Tests for somatic_marker_arousal function."""

    def test_baseline_arousal(self):
        """Should return 0 for baseline arousal (0.5)."""
        result = somatic_marker_arousal(0.5)
        assert result == 0.0

    def test_high_arousal(self):
        """Should return positive for high arousal."""
        result = somatic_marker_arousal(1.0)
        assert result == 2.0

    def test_low_arousal(self):
        """Should return negative for low arousal."""
        result = somatic_marker_arousal(0.0)
        assert result == -2.0

    def test_clamping_max(self):
        """Should clamp to +2 maximum."""
        result = somatic_marker_arousal(10.0)
        assert result == 2.0

    def test_clamping_min(self):
        """Should clamp to -2 minimum."""
        result = somatic_marker_arousal(-10.0)
        assert result == -2.0

    def test_linear_mapping(self):
        """Should map linearly between 0 and 1."""
        result_25 = somatic_marker_arousal(0.25)
        result_75 = somatic_marker_arousal(0.75)
        assert result_25 == -1.0
        assert result_75 == 1.0


class TestSomaticMarkerValence:
    """Tests for somatic_marker_valence function."""

    def test_positive_valence_high_arousal(self):
        """Should amplify positive valence with high arousal."""
        result = somatic_marker_valence(valence=1.0, arousal=1.0)
        # M = 2.0 * 1.0 * (0.5 + 1.0) = 3.0, clamped to 2.0
        assert result == 2.0

    def test_negative_valence_high_arousal(self):
        """Should amplify negative valence with high arousal."""
        result = somatic_marker_valence(valence=-1.0, arousal=1.0)
        # M = 2.0 * (-1.0) * (0.5 + 1.0) = -3.0, clamped to -2.0
        assert result == -2.0

    def test_zero_valence(self):
        """Should return 0 for neutral valence."""
        result = somatic_marker_valence(valence=0.0, arousal=1.0)
        assert result == 0.0

    def test_baseline_arousal(self):
        """Should use moderate amplification at baseline arousal."""
        result = somatic_marker_valence(valence=1.0, arousal=0.5)
        # M = 2.0 * 1.0 * (0.5 + 0.5) = 2.0, clamped to 2.0
        assert result == 2.0

    def test_clamping(self):
        """Should clamp to valid range."""
        result_high = somatic_marker_valence(1.0, 10.0)
        result_low = somatic_marker_valence(-1.0, 10.0)
        assert result_high == 2.0
        assert result_low == -2.0


class TestComputePrecisionWithSomaticMarker:
    """Tests for compute_precision_with_somatic_marker function."""

    def test_positive_marker(self):
        """Should increase precision for positive marker."""
        result = compute_precision_with_somatic_marker(
            pi_baseline=1.0,
            beta=0.3,
            M=1.0,
        )
        # pi_eff = 1.0 * exp(0.3 * 1.0) = 1.0 * 1.3499 ≈ 1.35
        assert result > 1.0

    def test_negative_marker(self):
        """Should decrease precision for negative marker."""
        result = compute_precision_with_somatic_marker(
            pi_baseline=1.0,
            beta=0.3,
            M=-1.0,
        )
        # pi_eff = 1.0 * exp(0.3 * -1.0) = 1.0 * 0.7408 ≈ 0.74
        assert result < 1.0

    def test_zero_marker(self):
        """Should return baseline for zero marker."""
        result = compute_precision_with_somatic_marker(
            pi_baseline=1.0,
            beta=0.3,
            M=0.0,
        )
        assert result == 1.0

    def test_zero_beta(self):
        """Should return baseline when beta is zero."""
        result = compute_precision_with_somatic_marker(
            pi_baseline=2.0,
            beta=0.0,
            M=2.0,
        )
        assert result == 2.0

    def test_clamping_max(self):
        """Should clamp to pi_max."""
        result = compute_precision_with_somatic_marker(
            pi_baseline=500.0,
            beta=1.0,
            M=2.0,
            pi_min=1e-4,
            pi_max=1000.0,
        )
        # 500 * exp(2.0) = 500 * 7.389 = 3694 > 1000, so should be clamped
        assert result == 1000.0

    def test_clamping_min(self):
        """Should clamp to pi_min."""
        result = compute_precision_with_somatic_marker(
            pi_baseline=0.01,
            beta=3.0,
            M=-2.0,
            pi_min=1e-4,
            pi_max=1000.0,
        )
        # 0.01 * exp(-6.0) = 0.01 * 0.00248 = 0.0000248 < 1e-4, so should be clamped
        assert result == 1e-4


class TestComputeSomaticGain:
    """Tests for compute_somatic_gain function."""

    def test_positive_marker(self):
        """Should return gain > 1 for positive marker."""
        result = compute_somatic_gain(M=1.0, beta=0.3)
        expected = np.exp(0.3 * 1.0)
        assert pytest.approx(result, rel=1e-7) == expected
        assert result > 1.0

    def test_negative_marker(self):
        """Should return gain < 1 for negative marker."""
        result = compute_somatic_gain(M=-1.0, beta=0.3)
        expected = np.exp(0.3 * -1.0)
        assert pytest.approx(result, rel=1e-7) == expected
        assert result < 1.0

    def test_zero_marker(self):
        """Should return 1 for zero marker."""
        result = compute_somatic_gain(M=0.0, beta=0.3)
        assert result == 1.0

    def test_gain_range(self):
        """Should return gain in expected range for M in [-2, 2]."""
        gain_max = compute_somatic_gain(M=2.0, beta=0.3)
        gain_min = compute_somatic_gain(M=-2.0, beta=0.3)
        assert gain_max == np.exp(0.6)
        assert gain_min == np.exp(-0.6)


class TestUpdateSomaticMarkerEuler:
    """Tests for update_somatic_marker_euler function."""

    def test_approach_target(self):
        """Should approach target arousal over time."""
        M = 0.0  # Start at baseline
        tau_M = 500.0
        dt = 1.0

        # Multiple steps toward high arousal target
        for _ in range(1000):
            M = update_somatic_marker_euler(M, arousal_target=1.0, tau_M=tau_M, dt=dt)

        # Should be closer to +2.0 (target marker for arousal=1.0)
        assert M > 0.0

    def test_decay_to_target(self):
        """Should decay toward target from above."""
        M = 2.0  # Start high
        tau_M = 500.0
        dt = 1.0

        # Multiple steps toward low arousal target
        for _ in range(1000):
            M = update_somatic_marker_euler(M, arousal_target=0.0, tau_M=tau_M, dt=dt)

        # Should be closer to -2.0 (target marker for arousal=0.0)
        assert M < 2.0

    def test_clamping(self):
        """Should clamp to valid range."""
        # Even with extreme target, should stay clamped
        M = 0.0
        for _ in range(10000):
            M = update_somatic_marker_euler(M, arousal_target=10.0, tau_M=10.0, dt=1.0)

        assert pytest.approx(M, abs=1e-10) == 2.0  # Clamped to max

    def test_no_change_at_target(self):
        """Should not change when already at target."""
        M_target = somatic_marker_arousal(0.5)  # Should be 0.0
        result = update_somatic_marker_euler(
            M_target, arousal_target=0.5, tau_M=500.0, dt=1.0
        )
        assert result == M_target

    def test_time_constant_effect(self):
        """Should change faster with smaller tau_M."""
        M_slow = update_somatic_marker_euler(
            0.0, arousal_target=1.0, tau_M=1000.0, dt=100.0
        )
        M_fast = update_somatic_marker_euler(
            0.0, arousal_target=1.0, tau_M=100.0, dt=100.0
        )

        # Faster time constant should move further
        assert abs(M_fast) > abs(M_slow)
