"""Comprehensive unit tests for core/precision.py module.

Tests cover:
- clamp function
- compute_precision function
- EMA update functions (mean and variance)
- Neuromodulator gain functions
- Precision coupling ODE
- Euler integration for precision
"""

from __future__ import annotations

import pytest
from typing import Callable

from core.precision import (
    clamp,
    compute_precision,
    update_mean_ema,
    update_variance_ema,
    apply_ach_gain,
    apply_ne_gain,
    apply_dopamine_bias_to_error,
    compute_interoceptive_precision_exponential,
    precision_coupling_ode_core,
    update_precision_euler,
)


class TestClamp:
    """Tests for clamp function."""

    def test_clamp_within_range(self):
        """Should return value when within range."""
        assert clamp(5.0, 0.0, 10.0) == 5.0
        assert clamp(-3.0, -10.0, 10.0) == -3.0
        assert clamp(0.5, 0.0, 1.0) == 0.5

    def test_clamp_below_range(self):
        """Should return lower bound when below range."""
        assert clamp(-5.0, 0.0, 10.0) == 0.0
        assert clamp(-100.0, -10.0, 10.0) == -10.0

    def test_clamp_above_range(self):
        """Should return upper bound when above range."""
        assert clamp(15.0, 0.0, 10.0) == 10.0
        assert clamp(100.0, -10.0, 10.0) == 10.0

    def test_clamp_at_boundaries(self):
        """Should handle values at exact boundaries."""
        assert clamp(0.0, 0.0, 10.0) == 0.0
        assert clamp(10.0, 0.0, 10.0) == 10.0

    def test_clamp_invalid_bounds(self):
        """Should raise ValueError when lower > upper."""
        with pytest.raises(ValueError, match="lower must be <= upper"):
            clamp(5.0, 10.0, 0.0)

    def test_clamp_equal_bounds(self):
        """Should handle equal bounds."""
        assert clamp(5.0, 3.0, 3.0) == 3.0


class TestComputePrecision:
    """Tests for compute_precision function."""

    def test_compute_precision_basic(self):
        """Should compute precision from variance."""
        # Due to default eps=1e-8, use approximate comparison
        assert pytest.approx(compute_precision(1.0), rel=1e-5) == 1.0
        assert pytest.approx(compute_precision(0.25), rel=1e-5) == 4.0
        assert pytest.approx(compute_precision(4.0), rel=1e-5) == 0.25

    def test_compute_precision_with_epsilon(self):
        """Should handle small variances with epsilon."""
        result = compute_precision(1e-10, eps=1e-8, pi_min=1e-8, pi_max=1e10)
        expected = 1.0 / (1e-10 + 1e-8)
        assert pytest.approx(result, rel=1e-3) == expected

    def test_compute_precision_clamping(self):
        """Should clamp to pi_min and pi_max."""
        # Very small variance should be clamped to pi_max
        result_small = compute_precision(1e-10, pi_min=1e-4, pi_max=1e4)
        assert result_small == 1e4

        # Very large variance should be clamped to pi_min
        result_large = compute_precision(1e10, pi_min=1e-4, pi_max=1e4)
        assert result_large == 1e-4

    def test_compute_precision_zero_variance(self):
        """Should handle zero variance."""
        result = compute_precision(0.0)
        # Should use eps to avoid division by zero
        assert result > 0

    def test_compute_precision_negative_variance(self):
        """Should handle negative variance by treating as zero."""
        result = compute_precision(-1.0)
        # Should treat negative as zero and use eps
        assert result > 0


class TestUpdateMeanEMA:
    """Tests for update_mean_ema function."""

    def test_ema_basic(self):
        """Should compute EMA update correctly."""
        prev_mean = 1.0
        z = 2.0
        alpha = 0.5
        result = update_mean_ema(prev_mean, z, alpha)
        expected = (1 - 0.5) * 1.0 + 0.5 * 2.0
        assert result == expected

    def test_ema_alpha_one(self):
        """Should return current value when alpha=1."""
        result = update_mean_ema(1.0, 2.0, 1.0)
        assert result == 2.0

    def test_ema_alpha_near_zero(self):
        """Should stay near previous mean when alpha is small."""
        result = update_mean_ema(1.0, 2.0, 0.01)
        assert 0.99 < result < 1.02

    def test_ema_invalid_alpha(self):
        """Should raise ValueError for invalid alpha."""
        with pytest.raises(ValueError, match="alpha must be in"):
            update_mean_ema(1.0, 2.0, 0.0)

        with pytest.raises(ValueError, match="alpha must be in"):
            update_mean_ema(1.0, 2.0, 1.5)

        with pytest.raises(ValueError, match="alpha must be in"):
            update_mean_ema(1.0, 2.0, -0.1)


class TestUpdateVarianceEMA:
    """Tests for update_variance_ema function."""

    def test_variance_ema_basic(self):
        """Should compute variance EMA correctly."""
        prev_sigma2 = 1.0
        z = 2.0
        mu = 1.5
        alpha = 0.5
        result = update_variance_ema(prev_sigma2, z, mu, alpha)
        expected = (1 - 0.5) * 1.0 + 0.5 * (2.0 - 1.5) ** 2
        assert result == expected

    def test_variance_ema_zero_deviation(self):
        """Should decrease variance when z equals mean."""
        prev_sigma2 = 1.0
        z = 1.5
        mu = 1.5
        alpha = 0.5
        result = update_variance_ema(prev_sigma2, z, mu, alpha)
        expected = 0.5 * 1.0 + 0.5 * 0.0
        assert result == expected

    def test_variance_ema_invalid_alpha(self):
        """Should raise ValueError for invalid alpha."""
        with pytest.raises(ValueError, match="alpha must be in"):
            update_variance_ema(1.0, 2.0, 1.5, 0.0)


class TestApplyAchGain:
    """Tests for apply_ach_gain function."""

    def test_ach_gain_basic(self):
        """Should apply ACh gain correctly."""
        assert apply_ach_gain(1.0, 1.5) == 1.5
        assert apply_ach_gain(2.0, 0.5) == 1.0
        assert apply_ach_gain(0.0, 1.0) == 0.0

    def test_ach_gain_negative(self):
        """Should handle negative precision and gain."""
        assert apply_ach_gain(-1.0, 1.0) == -1.0
        assert apply_ach_gain(1.0, -0.5) == -0.5


class TestApplyNeGain:
    """Tests for apply_ne_gain function."""

    def test_ne_gain_basic(self):
        """Should apply NE gain correctly."""
        assert apply_ne_gain(1.0, 1.5) == 1.5
        assert apply_ne_gain(2.0, 0.5) == 1.0

    def test_ne_gain_zero(self):
        """Should handle zero gain."""
        assert apply_ne_gain(5.0, 0.0) == 0.0


class TestApplyDopamineBiasToError:
    """Tests for apply_dopamine_bias_to_error function."""

    def test_dopamine_bias_basic(self):
        """Should apply dopamine bias correctly."""
        assert apply_dopamine_bias_to_error(1.0, 0.5) == 1.5
        assert apply_dopamine_bias_to_error(-1.0, 0.5) == -0.5
        assert apply_dopamine_bias_to_error(1.0, -0.5) == 0.5

    def test_dopamine_bias_zero(self):
        """Should return original when beta is zero."""
        assert apply_dopamine_bias_to_error(1.5, 0.0) == 1.5


class TestComputeInteroceptivePrecisionExponential:
    """Tests for compute_interoceptive_precision_exponential function."""

    def test_exponential_modulation_positive_m(self):
        """Should increase precision for positive M."""
        result = compute_interoceptive_precision_exponential(1.0, 0.3, 1.0)
        assert result > 1.0  # exp(0.3 * 1.0) > 1

    def test_exponential_modulation_negative_m(self):
        """Should decrease precision for negative M."""
        result = compute_interoceptive_precision_exponential(1.0, 0.3, -1.0)
        assert result < 1.0  # exp(0.3 * -1.0) < 1

    def test_exponential_modulation_zero_m(self):
        """Should return baseline for M=0."""
        result = compute_interoceptive_precision_exponential(1.0, 0.3, 0.0)
        assert result == 1.0

    def test_exponential_modulation_clamping(self):
        """Should clamp results to bounds."""
        # Large positive M should be clamped to pi_max
        # exp(2.0 * 10.0) = exp(20) ≈ 4.85e8 > 1e4
        result_large = compute_interoceptive_precision_exponential(
            1.0, 2.0, 10.0, pi_min=1e-4, pi_max=1e4
        )
        assert result_large == 1e4

        # Large negative M should be clamped to pi_min
        # exp(2.0 * -10.0) = exp(-20) ≈ 2.06e-9 < 1e-4
        result_small = compute_interoceptive_precision_exponential(
            1.0, 2.0, -10.0, pi_min=1e-4, pi_max=1e4
        )
        assert result_small == 1e-4


class TestPrecisionCouplingODECore:
    """Tests for precision_coupling_ode_core function."""

    def test_precision_coupling_basic(self):
        """Should compute precision coupling correctly."""
        result = precision_coupling_ode_core(
            pi_ell=1.0,
            tau_pi=1000.0,
            epsilon_ell=0.5,
            alpha_gain=0.1,
            pi_ell_plus_1=1.5,
            epsilon_ell_minus_1=0.3,
            C_down=0.1,
            C_up=0.05,
        )

        # Expected: -1.0/1000 + 0.1*0.5 + 0.1*(1.5-1.0) + 0.05*0.3
        expected = -0.001 + 0.05 + 0.05 + 0.015
        assert pytest.approx(result, rel=1e-6) == expected

    def test_precision_coupling_no_neighbors(self):
        """Should handle boundary levels (no neighbors)."""
        # Top level (no pi_ell_plus_1)
        result_top = precision_coupling_ode_core(
            pi_ell=1.0,
            tau_pi=1000.0,
            epsilon_ell=0.5,
            alpha_gain=0.1,
            pi_ell_plus_1=None,
            epsilon_ell_minus_1=0.3,
            C_down=0.1,
            C_up=0.05,
        )

        # Bottom level (no epsilon_ell_minus_1)
        result_bottom = precision_coupling_ode_core(
            pi_ell=1.0,
            tau_pi=1000.0,
            epsilon_ell=0.5,
            alpha_gain=0.1,
            pi_ell_plus_1=1.5,
            epsilon_ell_minus_1=None,
            C_down=0.1,
            C_up=0.05,
        )

        # Top level should have no top-down coupling
        assert result_top == -0.001 + 0.05 + 0 + 0.015

        # Bottom level should have no bottom-up coupling
        assert result_bottom == -0.001 + 0.05 + 0.05 + 0

    def test_precision_coupling_with_psi(self) -> None:
        """Should apply psi function to bottom-up error."""
        psi: Callable[[float], float] = lambda x: x**2

        result = precision_coupling_ode_core(
            pi_ell=1.0,
            tau_pi=1000.0,
            epsilon_ell=0.5,
            alpha_gain=0.1,
            pi_ell_plus_1=None,
            epsilon_ell_minus_1=2.0,
            C_down=0.1,
            C_up=0.05,
            psi=psi,
        )

        # Expected: -0.001 + 0.05 + 0 + 0.05*(2.0^2)
        expected = -0.001 + 0.05 + 0 + 0.05 * 4.0
        assert pytest.approx(result, rel=1e-6) == expected


class TestUpdatePrecisionEuler:
    """Tests for update_precision_euler function."""

    def test_euler_update_basic(self):
        """Should update precision correctly."""
        result = update_precision_euler(1.0, 0.1, 0.5)
        expected = 1.0 + 0.5 * 0.1
        assert result == expected

    def test_euler_update_clamping(self):
        """Should clamp to pi_min and pi_max."""
        # Update that would exceed pi_max
        result_max = update_precision_euler(1.0, 100.0, 0.5, pi_min=1e-4, pi_max=10.0)
        assert result_max == 10.0

        # Update that would go below pi_min
        result_min = update_precision_euler(0.01, -1.0, 0.5, pi_min=1e-4, pi_max=10.0)
        assert result_min == 1e-4

    def test_euler_update_negative_precision(self):
        """Should handle updates that would make precision negative."""
        result = update_precision_euler(0.1, -1.0, 0.5, pi_min=0.01, pi_max=100.0)
        assert result == 0.01
