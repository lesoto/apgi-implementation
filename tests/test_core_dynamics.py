"""Comprehensive unit tests for core/dynamics.py module.

Tests cover:
- signal_drift function
- update_signal_ode function
- compute_precision_coupled_noise_std function
- update_prediction function
- update_threshold_ode function
"""

from __future__ import annotations

import numpy as np
import pytest

from core.dynamics import (
    compute_precision_coupled_noise_std,
    signal_drift,
    update_prediction,
    update_signal_ode,
    update_threshold_ode,
)


class TestSignalDrift:
    """Tests for signal_drift function."""

    def test_basic_drift(self):
        """Should compute deterministic drift correctly."""
        result = signal_drift(
            S=1.0,
            phi_e=0.5,
            phi_i=0.5,
            pi_e=2.0,
            pi_i=1.0,
            tau_s=5.0,
        )
        # drift = -1.0/5.0 + 2.0*0.5 + 1.0*0.5 = -0.2 + 1.0 + 0.5 = 1.3
        expected = -0.2 + 1.0 + 0.5
        assert pytest.approx(result, rel=1e-7) == expected

    def test_invalid_tau_s(self):
        """Should raise ValueError for non-positive tau_s."""
        with pytest.raises(ValueError, match="tau_s must be > 0"):
            signal_drift(1.0, 0.5, 0.3, 2.0, 1.0, tau_s=0.0)

        with pytest.raises(ValueError, match="tau_s must be > 0"):
            signal_drift(1.0, 0.5, 0.3, 2.0, 1.0, tau_s=-1.0)

    def test_zero_dopamine(self):
        """Should compute drift with different phi_i values."""
        result1 = signal_drift(1.0, 0.5, 0.3, 2.0, 1.0, tau_s=5.0)
        result2 = signal_drift(1.0, 0.5, 0.5, 2.0, 1.0, tau_s=5.0)
        assert result1 != result2

    def test_negative_drift(self):
        """Should return negative drift when S is large."""
        result = signal_drift(
            S=100.0,
            phi_e=0.0,
            phi_i=0.0,
            pi_e=1.0,
            pi_i=1.0,
            tau_s=5.0,
        )
        assert result < 0  # Strong decay term dominates


class TestUpdateSignalODE:
    """Tests for update_signal_ode function."""

    def test_basic_update(self):
        """Should update signal with deterministic and stochastic parts."""
        np.random.seed(42)
        result = update_signal_ode(
            S=1.0,
            phi_e=0.5,
            phi_i=0.5,
            pi_e=2.0,
            pi_i=1.0,
            tau_s=5.0,
            dt=1.0,
            noise_std=0.01,
        )
        # Result should be close to S + drift*dt
        # (exact value depends on random noise)
        assert 0.5 < result < 3.0  # Reasonable range

    def test_invalid_tau_s(self):
        """Should raise ValueError for non-positive tau_s."""
        with pytest.raises(ValueError, match="tau_s must be > 0"):
            update_signal_ode(1.0, 0.5, 0.3, 2.0, 1.0, tau_s=0.0)

    def test_small_time_step(self):
        """Should make smaller changes with smaller dt."""
        np.random.seed(42)
        result_small_dt = update_signal_ode(
            S=1.0,
            phi_e=0.5,
            phi_i=0.5,
            pi_e=2.0,
            pi_i=1.0,
            tau_s=5.0,
            dt=0.1,
            noise_std=0.01,
        )

        np.random.seed(42)
        result_large_dt = update_signal_ode(
            S=1.0,
            phi_e=0.5,
            phi_i=0.5,
            pi_e=2.0,
            pi_i=1.0,
            tau_s=5.0,
            dt=1.0,
            noise_std=0.01,
        )

        # Changes should be smaller with smaller dt (deterministic part)
        drift_small_dt = result_small_dt - 1.0
        drift_large_dt = result_large_dt - 1.0
        assert abs(drift_small_dt) < abs(drift_large_dt)

    def test_zero_noise(self):
        """Should be deterministic when noise_std=0."""
        np.random.seed(42)
        result1 = update_signal_ode(
            S=1.0,
            phi_e=0.5,
            phi_i=0.5,
            pi_e=2.0,
            pi_i=1.0,
            tau_s=5.0,
            dt=1.0,
            noise_std=0.0,
        )

        np.random.seed(999)  # Different seed
        result2 = update_signal_ode(
            S=1.0,
            phi_e=0.5,
            phi_i=0.5,
            pi_e=2.0,
            pi_i=1.0,
            tau_s=5.0,
            dt=1.0,
            noise_std=0.0,
        )

        assert result1 == result2


class TestComputePrecisionCoupledNoiseStd:
    """Tests for compute_precision_coupled_noise_std function."""

    def test_basic_computation(self):
        """Should compute noise std correctly."""
        result = compute_precision_coupled_noise_std(pi_e_eff=4.0, pi_i_eff=4.0)
        # σ = 1 / sqrt(4.0 + 4.0) = 1 / sqrt(8.0) = 1 / (2*sqrt(2))
        expected = 1.0 / np.sqrt(8.0)
        assert pytest.approx(result, rel=1e-7) == expected

    def test_high_precision_low_noise(self):
        """Should return low noise for high precision."""
        result = compute_precision_coupled_noise_std(pi_e_eff=100.0, pi_i_eff=100.0)
        assert result < 0.1

    def test_low_precision_high_noise(self):
        """Should return high noise for low precision."""
        result = compute_precision_coupled_noise_std(pi_e_eff=0.01, pi_i_eff=0.01)
        assert result > 5.0

    def test_zero_precision(self):
        """Should return default noise when precision is zero."""
        result = compute_precision_coupled_noise_std(pi_e_eff=0.0, pi_i_eff=0.0)
        assert result == 1.0  # Default value

    def test_negative_precision(self):
        """Should handle negative precision gracefully."""
        # Should not crash but return reasonable value
        result = compute_precision_coupled_noise_std(pi_e_eff=-1.0, pi_i_eff=5.0)
        expected = 1.0 / np.sqrt(4.0)  # Only positive precision used
        assert pytest.approx(result, rel=1e-7) == expected


class TestUpdatePrediction:
    """Tests for update_prediction function."""

    def test_basic_update(self):
        """Should update prediction correctly."""
        result = update_prediction(
            x_hat=1.0,
            epsilon=0.5,
            pi=2.0,
            kappa=0.1,
        )
        # x_hat_new = 1.0 + 0.1 * 2.0 * 0.5 = 1.0 + 0.1 = 1.1
        expected = 1.1
        assert result == expected

    def test_zero_error(self):
        """Should not change prediction when error is zero."""
        result = update_prediction(
            x_hat=1.0,
            epsilon=0.0,
            pi=2.0,
            kappa=0.1,
        )
        assert result == 1.0

    def test_zero_kappa(self):
        """Should not change prediction when kappa is zero."""
        result = update_prediction(
            x_hat=1.0,
            epsilon=0.5,
            pi=2.0,
            kappa=0.0,
        )
        assert result == 1.0

    def test_high_precision(self):
        """Should make larger updates with higher precision."""
        result_low_pi = update_prediction(x_hat=1.0, epsilon=0.5, pi=0.1, kappa=0.1)
        result_high_pi = update_prediction(x_hat=1.0, epsilon=0.5, pi=10.0, kappa=0.1)
        change_low = abs(result_low_pi - 1.0)
        change_high = abs(result_high_pi - 1.0)
        assert change_high > change_low


class TestUpdateThresholdODE:
    """Tests for update_threshold_ode function."""

    def test_basic_update(self):
        """Should update threshold correctly."""
        np.random.seed(42)
        result = update_threshold_ode(
            theta=1.0,
            theta_base=1.0,
            C=1.5,
            V=1.0,
            tau_theta=1000.0,
            eta=0.1,
            dt=1.0,
            noise_std=0.01,
        )
        # At baseline with C > V, threshold should increase
        assert result > 1.0

    def test_invalid_tau_theta(self):
        """Should raise ValueError for non-positive tau_theta."""
        with pytest.raises(ValueError, match="tau_theta must be > 0"):
            update_threshold_ode(theta=1.0, theta_base=1.0, C=1.5, V=1.0, tau_theta=0.0, eta=0.1)

        with pytest.raises(ValueError, match="tau_theta must be > 0"):
            update_threshold_ode(theta=1.0, theta_base=1.0, C=1.5, V=1.0, tau_theta=-1.0, eta=0.1)

    def test_mean_reversion(self):
        """Should revert to baseline over time."""
        np.random.seed(42)
        theta = 2.0  # Above baseline
        theta_base = 1.0

        # Run multiple steps
        for _ in range(100):
            theta = update_threshold_ode(
                theta=theta,
                theta_base=theta_base,
                C=1.0,
                V=1.0,
                tau_theta=100.0,
                eta=0.0,
                dt=1.0,
                noise_std=0.0,
            )

        # Should be closer to baseline
        assert abs(theta - theta_base) < abs(2.0 - theta_base)

    def test_cost_value_mismatch(self):
        """Should increase threshold when C > V."""
        np.random.seed(42)
        result = update_threshold_ode(
            theta=1.0,
            theta_base=1.0,
            C=2.0,
            V=1.0,
            tau_theta=1000.0,
            eta=0.1,
            dt=1.0,
            noise_std=0.0,
        )
        assert result > 1.0

    def test_value_greater_than_cost(self):
        """Should decrease threshold when V > C."""
        np.random.seed(42)
        result = update_threshold_ode(
            theta=1.0,
            theta_base=1.0,
            C=1.0,
            V=2.0,
            tau_theta=1000.0,
            eta=0.1,
            dt=1.0,
            noise_std=0.0,
        )
        assert result < 1.0
