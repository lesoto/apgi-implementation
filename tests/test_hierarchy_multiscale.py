"""Comprehensive unit tests for hierarchy/multiscale.py module.

Tests cover:
- build_timescales function
- update_multiscale_feature function
- multiscale_weights function
- aggregate_multiscale_signal function
- apply_reset_rule function
- phase_signal function
- modulate_threshold function
- bottom_up_cascade function
"""

from __future__ import annotations

import numpy as np
import pytest

from hierarchy.multiscale import (
    build_timescales,
    update_multiscale_feature,
    multiscale_weights,
    aggregate_multiscale_signal,
    apply_reset_rule,
    phase_signal,
    modulate_threshold,
    bottom_up_cascade,
)


class TestBuildTimescales:
    """Tests for build_timescales function."""

    def test_basic_build(self):
        """Should build timescales correctly."""
        result = build_timescales(tau0=10.0, k=2.0, n_levels=3)
        expected = np.array([10.0, 20.0, 40.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_single_level(self):
        """Should handle single level."""
        result = build_timescales(tau0=10.0, k=2.0, n_levels=1)
        assert len(result) == 1
        assert result[0] == 10.0

    def test_invalid_parameters(self):
        """Should raise ValueError for invalid parameters."""
        with pytest.raises(ValueError, match="tau0 must be > 0"):
            build_timescales(tau0=0, k=2.0, n_levels=3)

        with pytest.raises(ValueError, match="k must be > 1"):
            build_timescales(tau0=10.0, k=1.0, n_levels=3)

        with pytest.raises(ValueError, match="n_levels must be > 0"):
            build_timescales(tau0=10.0, k=2.0, n_levels=0)

    def test_negative_tau0_raises_error(self):
        """Should raise error for negative tau0."""
        with pytest.raises(ValueError, match="tau0 must be > 0"):
            build_timescales(tau0=-5.0, k=2.0, n_levels=3)

    def test_k_less_than_one_raises_error(self):
        """Should raise error for k < 1."""
        with pytest.raises(ValueError, match="k must be > 1"):
            build_timescales(tau0=10.0, k=0.5, n_levels=3)

    def test_negative_n_levels_raises_error(self):
        """Should raise error for negative n_levels."""
        with pytest.raises(ValueError, match="n_levels must be > 0"):
            build_timescales(tau0=10.0, k=2.0, n_levels=-1)


class TestUpdateMultiscaleFeature:
    """Tests for update_multiscale_feature function."""

    def test_basic_update(self):
        """Should update feature correctly."""
        result = update_multiscale_feature(phi_prev=0.5, z_t=1.0, tau_i=10.0)
        a = 1.0 / 10.0
        expected = (1 - a) * 0.5 + a * 1.0
        assert pytest.approx(result, rel=1e-7) == expected

    def test_long_timescale(self):
        """Should change slowly with long timescale."""
        result = update_multiscale_feature(phi_prev=0.5, z_t=1.0, tau_i=1000.0)
        a = 1.0 / 1000.0
        expected = (1 - a) * 0.5 + a * 1.0
        assert pytest.approx(result, rel=1e-7) == expected

    def test_invalid_tau(self):
        """Should raise ValueError for non-positive tau."""
        with pytest.raises(ValueError, match="tau_i must be > 0"):
            update_multiscale_feature(phi_prev=0.5, z_t=1.0, tau_i=0)

    def test_negative_tau_raises_error(self):
        """Should raise error for negative tau_i."""
        with pytest.raises(ValueError, match="tau_i must be > 0"):
            update_multiscale_feature(phi_prev=0.5, z_t=1.0, tau_i=-5.0)

    def test_zero_tau_raises_error(self):
        """Should raise error for zero tau_i."""
        with pytest.raises(ValueError, match="tau_i must be > 0"):
            update_multiscale_feature(phi_prev=0.5, z_t=1.0, tau_i=0.0)

    def test_small_tau(self):
        """Should handle very small tau."""
        result = update_multiscale_feature(phi_prev=0.5, z_t=1.0, tau_i=0.1)
        assert isinstance(result, float)

    def test_large_tau(self):
        """Should handle very large tau."""
        result = update_multiscale_feature(phi_prev=0.5, z_t=1.0, tau_i=10000.0)
        assert isinstance(result, float)


class TestMultiscaleWeights:
    """Tests for multiscale_weights function."""

    def test_basic_weights(self):
        """Should compute normalized weights."""
        result = multiscale_weights(n_levels=3, k=2.0)
        # Raw: [1, 0.5, 0.25], Sum = 1.75, Normalized: [0.571, 0.286, 0.143]
        assert len(result) == 3
        assert pytest.approx(result[0], rel=1e-3) == 1 / 1.75
        assert pytest.approx(result[1], rel=1e-3) == 0.5 / 1.75
        assert pytest.approx(result[2], rel=1e-3) == 0.25 / 1.75

    def test_weights_sum_to_one(self):
        """Should sum to one."""
        result = multiscale_weights(n_levels=5, k=1.6)
        assert pytest.approx(np.sum(result), rel=1e-7) == 1.0

    def test_single_level_weights(self):
        """Should handle single level."""
        result = multiscale_weights(n_levels=1, k=2.0)
        assert len(result) == 1
        assert result[0] == 1.0

    def test_large_k(self):
        """Should handle large k."""
        result = multiscale_weights(n_levels=3, k=10.0)
        assert len(result) == 3
        assert pytest.approx(np.sum(result), rel=1e-7) == 1.0


class TestAggregateMultiscaleSignal:
    """Tests for aggregate_multiscale_signal function."""

    def test_basic_aggregation(self):
        """Should aggregate signal correctly."""
        phi_values = np.array([1.0, 2.0, 3.0])
        pi_values = np.array([1.0, 1.0, 1.0])
        weights = np.array([0.5, 0.3, 0.2])

        result = aggregate_multiscale_signal(phi_values, pi_values, weights)
        # S = 0.5*1*1 + 0.3*1*2 + 0.2*1*3 = 0.5 + 0.6 + 0.6 = 1.7
        expected = 1.7
        assert pytest.approx(result, rel=1e-10) == expected

    def test_length_mismatch(self):
        """Should raise ValueError for mismatched lengths."""
        with pytest.raises(ValueError, match="must have same length"):
            aggregate_multiscale_signal(
                phi_values=np.array([1.0, 2.0]),
                pi_values=np.array([1.0]),
                weights=np.array([0.5, 0.5]),
            )

    def test_mismatched_phi_pi_lengths(self):
        """Should raise error for phi/pi length mismatch."""
        with pytest.raises(ValueError, match="must have same length"):
            aggregate_multiscale_signal(
                phi_values=np.array([1.0, 2.0, 3.0]),
                pi_values=np.array([1.0, 1.0]),
                weights=np.array([0.5, 0.3, 0.2]),
            )

    def test_mismatched_phi_weights_lengths(self):
        """Should raise error for phi/weights length mismatch."""
        with pytest.raises(ValueError, match="must have same length"):
            aggregate_multiscale_signal(
                phi_values=np.array([1.0, 2.0]),
                pi_values=np.array([1.0, 1.0, 1.0]),
                weights=np.array([0.5, 0.3, 0.2]),
            )

    def test_with_list_inputs(self):
        """Should accept list inputs."""
        phi_values = [1.0, 2.0, 3.0]
        pi_values = [1.0, 1.0, 1.0]
        weights = [0.5, 0.3, 0.2]

        result = aggregate_multiscale_signal(phi_values, pi_values, weights)
        expected = 1.7
        assert pytest.approx(result, rel=1e-10) == expected

    def test_with_negative_phi(self):
        """Should handle negative phi values."""
        phi_values = np.array([-1.0, -2.0, -3.0])
        pi_values = np.array([1.0, 1.0, 1.0])
        weights = np.array([0.5, 0.3, 0.2])

        result = aggregate_multiscale_signal(phi_values, pi_values, weights)
        # abs(phi) makes them positive
        expected = 1.7
        assert pytest.approx(result, rel=1e-10) == expected


class TestApplyResetRule:
    """Tests for apply_reset_rule function."""

    def test_basic_reset(self):
        """Should apply reset correctly."""
        S_new, theta_new = apply_reset_rule(S=2.0, theta=1.0, rho=0.5, delta=0.5)
        assert S_new == 1.0  # 2.0 * 0.5
        assert theta_new == 1.5  # 1.0 + 0.5

    def test_no_reset(self):
        """Should not reset with rho=1 and delta=0."""
        S_new, theta_new = apply_reset_rule(S=2.0, theta=1.0, rho=1.0, delta=0.0)
        assert S_new == 2.0
        assert theta_new == 1.0


class TestPhaseSignal:
    """Tests for phase_signal function."""

    def test_basic_phase(self):
        """Should compute phase correctly."""
        result = phase_signal(omega=1.0, t=1.0, phi0=0.0)
        assert result == 1.0

    def test_with_initial_phase(self):
        """Should include initial phase."""
        result = phase_signal(omega=1.0, t=1.0, phi0=0.5)
        assert result == 1.5

    def test_zero_omega(self):
        """Should handle zero omega."""
        result = phase_signal(omega=0.0, t=10.0, phi0=0.5)
        assert result == 0.5

    def test_zero_time(self):
        """Should handle zero time."""
        result = phase_signal(omega=1.0, t=0.0, phi0=0.5)
        assert result == 0.5

    def test_large_omega(self):
        """Should handle large omega."""
        result = phase_signal(omega=100.0, t=1.0, phi0=0.0)
        assert result == 100.0

    def test_negative_omega(self):
        """Should handle negative omega."""
        result = phase_signal(omega=-1.0, t=1.0, phi0=0.0)
        assert result == -1.0


class TestModulateThreshold:
    """Tests for modulate_threshold function."""

    def test_phase_modulation(self):
        """Should modulate threshold based on phase."""
        result = modulate_threshold(
            theta_0=1.0, pi_above=1.0, phi_above=0.0, k_down=0.1
        )
        # theta = 1.0 * (1 + 0.1 * 1.0 * 1.0) = 1.1
        assert pytest.approx(result, rel=1e-7) == 1.1

    def test_opposite_phase(self):
        """Should decrease threshold at opposite phase."""
        result = modulate_threshold(
            theta_0=1.0, pi_above=1.0, phi_above=np.pi, k_down=0.1
        )
        # theta = 1.0 * (1 + 0.1 * 1.0 * -1.0) = 0.9
        assert pytest.approx(result, rel=1e-7) == 0.9


class TestBottomUpCascade:
    """Tests for bottom_up_cascade function."""

    def test_superthreshold(self):
        """Should reduce threshold when S_lower > theta_lower."""
        result = bottom_up_cascade(theta=1.0, S_lower=1.5, theta_lower=1.0, k_up=0.2)
        assert result == 0.8

    def test_subthreshold(self):
        """Should not change threshold when S_lower <= theta_lower."""
        result = bottom_up_cascade(theta=1.0, S_lower=0.5, theta_lower=1.0, k_up=0.2)
        assert result == 1.0
