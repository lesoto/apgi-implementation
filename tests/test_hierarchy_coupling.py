"""Comprehensive unit tests for hierarchy/coupling.py module.

Tests cover:
- estimate_hierarchy_levels function
- precision_coupling_ode function
- phase_locked_threshold function
- bottom_up_threshold_cascade function
- update_phase_dynamics function
- HierarchicalPrecisionNetwork class
"""

from __future__ import annotations

import numpy as np
import pytest

from hierarchy.coupling import (
    estimate_hierarchy_levels,
    precision_coupling_ode,
    phase_locked_threshold,
    bottom_up_threshold_cascade,
    update_phase_dynamics,
    update_phase_kuramoto_full,
    HierarchicalPrecisionNetwork,
)


class TestEstimateHierarchyLevels:
    """Tests for estimate_hierarchy_levels function."""

    def test_basic_estimation(self):
        """Should estimate correct number of levels."""
        result = estimate_hierarchy_levels(
            tau_min=10.0,
            tau_max=1000.0,
            k=1.6,
        )
        expected = int(np.floor(np.log(100) / np.log(1.6))) + 1
        assert result == expected

    def test_invalid_parameters(self):
        """Should raise ValueError for invalid parameters."""
        with pytest.raises(ValueError):
            estimate_hierarchy_levels(tau_min=0, tau_max=100, k=1.6)

        with pytest.raises(ValueError):
            estimate_hierarchy_levels(tau_min=10, tau_max=0, k=1.6)

        with pytest.raises(ValueError):
            estimate_hierarchy_levels(tau_min=10, tau_max=100, k=1.0)

    def test_single_level(self):
        """Should return 1 when tau_min close to tau_max."""
        result = estimate_hierarchy_levels(tau_min=100, tau_max=101, k=1.6)
        assert result == 1


class TestPrecisionCouplingODE:
    """Tests for precision_coupling_ode function."""

    def test_basic_coupling(self):
        """Should compute precision coupling correctly."""
        result = precision_coupling_ode(
            pi_ell=1.0,
            tau_pi=1000.0,
            epsilon_ell=0.5,
            alpha_gain=0.1,
            pi_ell_plus_1=1.5,
            epsilon_ell_minus_1=0.3,
            C_down=0.1,
            C_up=0.05,
        )
        expected = -0.001 + 0.05 + 0.05 + 0.015
        assert pytest.approx(result, rel=1e-6) == expected

    def test_top_level(self):
        """Should handle top level (no higher level)."""
        result = precision_coupling_ode(
            pi_ell=1.0,
            tau_pi=1000.0,
            epsilon_ell=0.5,
            alpha_gain=0.1,
            pi_ell_plus_1=None,
            epsilon_ell_minus_1=0.3,
            C_down=0.1,
            C_up=0.05,
        )
        expected = -0.001 + 0.05 + 0 + 0.015
        assert pytest.approx(result, rel=1e-6) == expected

    def test_with_psi_function(self):
        """Should apply psi function to lower level error when provided."""

        # Define a simple psi function
        def psi_func(x):
            return x * 2.0

        result = precision_coupling_ode(
            pi_ell=1.0,
            tau_pi=1000.0,
            epsilon_ell=0.5,
            alpha_gain=0.1,
            pi_ell_plus_1=None,
            epsilon_ell_minus_1=0.3,
            C_down=0.1,
            C_up=0.05,
            psi=psi_func,
        )
        # With psi doubling the error, bottom-up should be 0.05 * 0.3 * 2 = 0.03
        expected = -0.001 + 0.05 + 0 + 0.03
        assert pytest.approx(result, rel=1e-6) == expected


class TestPhaseLockedThreshold:
    """Tests for phase_locked_threshold function."""

    def test_phase_modulation(self):
        """Should modulate threshold based on phase."""
        result = phase_locked_threshold(
            theta_0_ell=1.0,
            pi_ell_plus_1=1.0,
            phi_ell_plus_1=0.0,
            kappa_down=0.1,
        )
        expected = 1.1
        assert pytest.approx(result, rel=1e-6) == expected

    def test_opposite_phase(self):
        """Should decrease threshold at opposite phase."""
        result = phase_locked_threshold(
            theta_0_ell=1.0,
            pi_ell_plus_1=1.0,
            phi_ell_plus_1=np.pi,
            kappa_down=0.1,
        )
        expected = 0.9
        assert pytest.approx(result, rel=1e-6) == expected


class TestBottomUpThresholdCascade:
    """Tests for bottom_up_threshold_cascade function."""

    def test_superthreshold_boost(self):
        """Should reduce threshold when lower level is superthreshold."""
        result = bottom_up_threshold_cascade(
            theta_ell=1.0,
            S_ell_minus_1=1.5,
            theta_ell_minus_1=1.0,
            kappa_up=0.2,
        )
        expected = 0.8
        assert pytest.approx(result, rel=1e-6) == expected

    def test_subthreshold_no_boost(self):
        """Should not change threshold when lower level is subthreshold."""
        result = bottom_up_threshold_cascade(
            theta_ell=1.0,
            S_ell_minus_1=0.5,
            theta_ell_minus_1=1.0,
            kappa_up=0.2,
        )
        assert result == 1.0


class TestUpdatePhaseDynamics:
    """Tests for update_phase_dynamics function."""

    def test_free_rotation(self):
        """Should rotate phase based on natural frequency."""
        result = update_phase_dynamics(
            phi=0.0,
            omega=1.0,
            dt=1.0,
        )
        assert pytest.approx(result, rel=1e-6) == 1.0

    def test_wraparound(self):
        """Should wrap phase to [0, 2*pi]."""
        result = update_phase_dynamics(
            phi=2 * np.pi - 0.1,
            omega=1.0,
            dt=1.0,
        )
        assert result < 2 * np.pi
        assert result >= 0


class TestHierarchicalPrecisionNetwork:
    """Tests for HierarchicalPrecisionNetwork class."""

    def test_initialization(self):
        """Should initialize with correct parameters."""
        network = HierarchicalPrecisionNetwork(
            n_levels=3,
            tau_pi=1000.0,
            C_down=0.1,
            C_up=0.05,
        )
        assert network.n_levels == 3
        assert len(network.pi) == 3

    def test_initialization_with_custom_taus(self):
        """Should initialize with custom taus array."""
        custom_taus = np.array([10.0, 100.0, 1000.0])
        network = HierarchicalPrecisionNetwork(
            n_levels=3,
            tau_pi=1000.0,
            C_down=0.1,
            C_up=0.05,
            taus=custom_taus,
        )
        assert network.n_levels == 3
        assert np.array_equal(network.taus, custom_taus)

    def test_step(self):
        """Should update precision and phase."""
        network = HierarchicalPrecisionNetwork(n_levels=3)
        epsilon_new = np.array([0.1, 0.2, 0.3])
        pi_new, phi_new = network.step(epsilon_new, dt=1.0, alpha_gain=0.1)
        assert len(pi_new) == 3
        assert all(p > 0 for p in pi_new)

    def test_precision_non_negative(self):
        """Should keep precision non-negative."""
        network = HierarchicalPrecisionNetwork(n_levels=3)
        for _ in range(10):
            epsilon = np.array([-1.0, -1.0, -1.0])
            pi_new, _ = network.step(epsilon, dt=1.0)
        assert all(p >= 0.01 for p in pi_new)

    def test_compute_thresholds_with_bottom_up_cascade(self):
        """Should compute thresholds with bottom-up cascade."""
        network = HierarchicalPrecisionNetwork(n_levels=3)
        theta_0 = np.array([1.0, 1.0, 1.0])
        S_levels = np.array([1.5, 0.5, 0.2])
        thetas = network.compute_thresholds(
            theta_0=theta_0,
            S_levels=S_levels,
            kappa_down=0.1,
            kappa_up=0.2,  # Enable bottom-up cascade
        )
        assert len(thetas) == 3
        # Level 1 should be modulated by level 0's superthreshold signal
        assert thetas[1] != theta_0[1]


class TestPhaseLockedThresholdExtended:
    """Extended tests for phase_locked_threshold with phase_sensitivity."""

    def test_phase_sensitivity_scaling(self):
        """Should apply phase sensitivity scaling when != 1.0."""
        result_default = phase_locked_threshold(
            theta_0_ell=1.0,
            pi_ell_plus_1=1.0,
            phi_ell_plus_1=0.0,
            kappa_down=0.1,
            phase_sensitivity=1.0,
        )
        result_scaled = phase_locked_threshold(
            theta_0_ell=1.0,
            pi_ell_plus_1=1.0,
            phi_ell_plus_1=0.0,
            kappa_down=0.1,
            phase_sensitivity=2.0,  # Non-default value
        )
        # With phase_sensitivity=2.0, modulation should be amplified
        assert result_scaled != result_default

    def test_phase_sensitivity_less_than_one(self):
        """Should apply phase sensitivity scaling < 1.0."""
        result = phase_locked_threshold(
            theta_0_ell=1.0,
            pi_ell_plus_1=1.0,
            phi_ell_plus_1=np.pi / 2,
            kappa_down=0.1,
            phase_sensitivity=0.5,  # Dampening factor
        )
        assert result > 0


class TestUpdatePhaseKuramotoFull:
    """Tests for update_phase_kuramoto_full function."""

    def test_full_kuramoto_update(self):
        """Should update all phases with full coupling matrix."""
        phi_array = np.array([0.0, np.pi / 2, np.pi])
        omega_array = np.array([1.0, 1.1, 0.9])
        coupling_matrix = np.array(
            [
                [0.0, 0.1, 0.0],
                [0.1, 0.0, 0.1],
                [0.0, 0.1, 0.0],
            ]
        )
        phi_new = update_phase_kuramoto_full(
            phi_array, omega_array, coupling_matrix, dt=0.1
        )
        assert len(phi_new) == 3
        assert np.all(phi_new >= 0) and np.all(phi_new < 2 * np.pi)

    def test_full_kuramoto_with_noise(self):
        """Should update phases with noise term."""
        phi_array = np.array([0.0, np.pi / 2])
        omega_array = np.array([1.0, 1.1])
        coupling_matrix = np.array(
            [
                [0.0, 0.1],
                [0.1, 0.0],
            ]
        )
        rng = np.random.default_rng(42)
        phi_new = update_phase_kuramoto_full(
            phi_array, omega_array, coupling_matrix, dt=0.1, noise_std=0.01, rng=rng
        )
        assert len(phi_new) == 2

    def test_full_kuramoto_with_zero_noise(self):
        """Should update phases without noise when noise_std=0."""
        phi_array = np.array([0.0, np.pi / 2])
        omega_array = np.array([1.0, 1.1])
        coupling_matrix = np.array(
            [
                [0.0, 0.1],
                [0.1, 0.0],
            ]
        )
        phi_new = update_phase_kuramoto_full(
            phi_array, omega_array, coupling_matrix, dt=0.1, noise_std=0.0
        )
        assert len(phi_new) == 2

    def test_no_coupling(self):
        """Should work with zero coupling matrix."""
        phi_array = np.array([0.0, np.pi / 2])
        omega_array = np.array([1.0, 1.0])
        coupling_matrix = np.zeros((2, 2))
        phi_new = update_phase_kuramoto_full(
            phi_array, omega_array, coupling_matrix, dt=1.0
        )
        assert len(phi_new) == 2
        # Without coupling, each phase should advance by omega * dt
        expected_0 = (0.0 + 1.0 * 1.0) % (2 * np.pi)
        assert pytest.approx(phi_new[0], rel=1e-6) == expected_0
