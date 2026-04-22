"""Extended tests for oscillation/phase.py to achieve 100% coverage."""

import numpy as np
from oscillation.phase import (
    PhaseOscillatorNetwork,
    hierarchical_phase_coupling,
)


class TestPhaseOscillatorNetworkExtended:
    """Extended tests for PhaseOscillatorNetwork."""

    def test_network_with_custom_frequencies(self):
        """Should accept custom frequencies."""
        freqs = np.array([5.0, 10.0, 20.0])  # Hz
        network = PhaseOscillatorNetwork(
            n_levels=3,
            frequencies=freqs,
        )
        # omegas should be 2*pi*f
        expected_omegas = 2 * np.pi * freqs
        np.testing.assert_array_almost_equal(network.omegas, expected_omegas)

    def test_network_with_custom_coupling(self):
        """Should accept custom coupling matrix."""
        custom_K = np.array(
            [
                [0.0, 0.3, 0.0],
                [0.3, 0.0, 0.3],
                [0.0, 0.3, 0.0],
            ]
        )
        network = PhaseOscillatorNetwork(
            n_levels=3,
            coupling_matrix=custom_K,
        )
        np.testing.assert_array_equal(network.K, custom_K)

    def test_set_phases(self):
        """Should set phases manually."""
        network = PhaseOscillatorNetwork(n_levels=3)
        new_phases = np.array([0.0, np.pi, 2 * np.pi])
        network.set_phases(new_phases)
        np.testing.assert_array_almost_equal(
            network.phases, np.array([0.0, np.pi, 0.0])  # 2*pi wraps to 0
        )


class TestHierarchicalPhaseCoupling:
    """Tests for hierarchical_phase_coupling function."""

    def test_coupling_with_zero_matrix(self):
        """Should handle zero coupling matrix."""
        phases = np.array([0.0, np.pi / 2, np.pi])
        omegas = np.array([1.0, 1.0, 1.0])
        K = np.zeros((3, 3))
        result = hierarchical_phase_coupling(phases, omegas, K)
        # With zero coupling, all coupling sums should be 0
        np.testing.assert_array_almost_equal(result, np.zeros(3))
