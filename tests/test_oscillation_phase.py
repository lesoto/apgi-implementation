"""Comprehensive unit tests for oscillation/phase.py module.

Tests cover:
- compute_phase function
- update_phase_euler function
- phase_coupling_kuramoto function
- hierarchical_phase_coupling function
- nearest_neighbor_coupling_matrix function
- PhaseOscillatorNetwork class
"""

from __future__ import annotations

import numpy as np
import pytest

from oscillation.phase import (
    compute_phase,
    update_phase_euler,
    phase_coupling_kuramoto,
    hierarchical_phase_coupling,
    nearest_neighbor_coupling_matrix,
    PhaseOscillatorNetwork,
)


class TestComputePhase:
    """Tests for compute_phase function."""

    def test_basic_computation(self):
        """Should compute phase correctly."""
        result = compute_phase(t=1.0, omega=1.0, phi_0=0.0)
        assert result == 1.0

    def test_with_initial_phase(self):
        """Should include initial phase."""
        result = compute_phase(t=1.0, omega=1.0, phi_0=0.5)
        assert result == 1.5

    def test_wraparound(self):
        """Should wrap phase to [0, 2*pi]."""
        result = compute_phase(t=2 * np.pi + 0.5, omega=1.0, phi_0=0.0)
        assert pytest.approx(result, rel=1e-7) == 0.5


class TestUpdatePhaseEuler:
    """Tests for update_phase_euler function."""

    def test_basic_update(self):
        """Should update phase correctly."""
        result = update_phase_euler(phi=0.0, omega=1.0, dt=1.0)
        assert result == 1.0

    def test_with_coupling(self):
        """Should include coupling term."""
        result = update_phase_euler(phi=0.0, omega=1.0, dt=1.0, coupling_sum=0.5)
        # dphi = (1.0 + 0.5) * 1.0 = 1.5
        assert result == 1.5

    def test_wraparound(self):
        """Should wrap to [0, 2*pi]."""
        result = update_phase_euler(phi=2 * np.pi - 0.1, omega=1.0, dt=1.0)
        assert result < 2 * np.pi


class TestPhaseCouplingKuramoto:
    """Tests for phase_coupling_kuramoto function."""

    def test_same_phase(self):
        """Should return 0 when phases are equal."""
        result = phase_coupling_kuramoto(phi_i=1.0, phi_j=1.0, K_ij=0.5, omega_i=1.0)
        # K * sin(0) = 0
        assert result == 0.0

    def test_phase_difference(self):
        """Should compute coupling based on phase difference."""
        result = phase_coupling_kuramoto(
            phi_i=0.0, phi_j=np.pi / 2, K_ij=0.5, omega_i=1.0
        )
        # K * sin(pi/2) = 0.5
        assert pytest.approx(result, rel=1e-7) == 0.5

    def test_negative_coupling(self):
        """Should return negative for opposite phase."""
        result = phase_coupling_kuramoto(
            phi_i=0.0, phi_j=-np.pi / 2, K_ij=0.5, omega_i=1.0
        )
        # K * sin(-pi/2) = -0.5
        assert pytest.approx(result, rel=1e-7) == -0.5


class TestNearestNeighborCouplingMatrix:
    """Tests for nearest_neighbor_coupling_matrix function."""

    def test_basic_matrix(self):
        """Should create correct coupling matrix."""
        K = nearest_neighbor_coupling_matrix(n_levels=3, K_up=0.1, K_down=0.2)

        assert K.shape == (3, 3)
        assert K[0, 1] == 0.2  # Downward coupling from level 1 to 0
        assert K[1, 0] == 0.1  # Upward coupling from level 0 to 1
        assert K[1, 2] == 0.2  # Downward coupling from level 2 to 1
        assert K[2, 1] == 0.1  # Upward coupling from level 1 to 2
        assert K[0, 0] == 0.0  # No self-coupling

    def test_single_level(self):
        """Should handle single level."""
        K = nearest_neighbor_coupling_matrix(n_levels=1, K_up=0.1, K_down=0.2)
        assert K.shape == (1, 1)
        assert K[0, 0] == 0.0


class TestHierarchicalPhaseCoupling:
    """Tests for hierarchical_phase_coupling function."""

    def test_basic_coupling(self):
        """Should compute coupling sums correctly."""
        phases = np.array([0.0, np.pi / 2])
        omegas = np.array([1.0, 1.0])
        K = np.array([[0.0, 0.1], [0.1, 0.0]])

        result = hierarchical_phase_coupling(phases, omegas, K)

        assert len(result) == 2
        # Oscillator 0 receives from 1: 0.1 * sin(pi/2) = 0.1
        assert pytest.approx(result[0], rel=1e-7) == 0.1
        # Oscillator 1 receives from 0: 0.1 * sin(-pi/2) = -0.1
        assert pytest.approx(result[1], rel=1e-7) == -0.1


class TestPhaseOscillatorNetwork:
    """Tests for PhaseOscillatorNetwork class."""

    def test_initialization(self):
        """Should initialize correctly."""
        network = PhaseOscillatorNetwork(n_levels=3)
        assert network.n_levels == 3
        assert len(network.phases) == 3
        assert len(network.omegas) == 3

    def test_step(self):
        """Should update phases."""
        network = PhaseOscillatorNetwork(n_levels=3)
        initial_phases = network.phases.copy()

        new_phases = network.step(dt=0.01)

        # Phases should have changed due to natural frequency
        assert not np.array_equal(new_phases, initial_phases)

    def test_get_phases(self):
        """Should return copy of phases."""
        network = PhaseOscillatorNetwork(n_levels=3)
        phases1 = network.get_phases()
        phases2 = network.get_phases()

        # Should be copies, not same object
        assert phases1 is not phases2
        np.testing.assert_array_equal(phases1, phases2)

    def test_set_phases(self):
        """Should set phases correctly."""
        network = PhaseOscillatorNetwork(n_levels=3)
        network.set_phases(np.array([0.5, 1.0, 1.5]))

        np.testing.assert_array_equal(network.phases, np.array([0.5, 1.0, 1.5]))

    def test_phase_wrapping(self):
        """Should wrap phases to [0, 2*pi]."""
        network = PhaseOscillatorNetwork(n_levels=3)
        network.set_phases(np.array([2 * np.pi + 0.5, 4 * np.pi, -0.5]))

        # Should be wrapped
        assert network.phases[0] == pytest.approx(0.5, abs=1e-10)
        assert network.phases[1] == pytest.approx(0.0, abs=1e-10)
        assert network.phases[2] == pytest.approx(2 * np.pi - 0.5, abs=1e-10)
