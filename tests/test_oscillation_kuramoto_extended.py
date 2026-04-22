"""Extended tests for oscillation/kuramoto.py to achieve 100% coverage."""

import numpy as np
from oscillation.kuramoto import (
    KuramotoOscillators,
    HierarchicalKuramotoSystem,
)


class TestKuramotoOscillatorsExtended:
    """Extended tests for KuramotoOscillators."""

    def test_oscillators_with_custom_frequencies(self):
        """Should accept custom frequencies array."""
        freqs = np.array([5.0, 10.0, 20.0])  # Hz
        osc = KuramotoOscillators(
            n_levels=3,
            frequencies=freqs,
        )
        # omegas should be 2*pi*f/1000 (convert Hz to rad/ms)
        expected_omegas = 2 * np.pi * freqs / 1000.0
        np.testing.assert_array_almost_equal(osc.omegas, expected_omegas)

    def test_oscillators_with_custom_coupling(self):
        """Should accept custom coupling matrix."""
        custom_K = np.array(
            [
                [0.0, 0.2, 0.0],
                [0.2, 0.0, 0.2],
                [0.0, 0.2, 0.0],
            ]
        )
        osc = KuramotoOscillators(
            n_levels=3,
            coupling_matrix=custom_K,
        )
        np.testing.assert_array_equal(osc.K, custom_K)

    def test_get_phases_returns_copy(self):
        """Should return a copy of phases."""
        osc = KuramotoOscillators(n_levels=3)
        phases = osc.get_phases()
        phases[0] = 999.0  # Modify returned array
        # Original should not be affected
        assert osc.phases[0] != 999.0

    def test_set_phases(self):
        """Should set phases manually."""
        osc = KuramotoOscillators(n_levels=3)
        new_phases = np.array([0.0, np.pi, 2 * np.pi])
        osc.set_phases(new_phases)
        np.testing.assert_array_almost_equal(
            osc.phases, np.array([0.0, np.pi, 0.0])  # 2*pi wraps to 0
        )

    def test_history_empty(self):
        """Should handle empty history."""
        osc = KuramotoOscillators(n_levels=3)
        # Before any steps, history is empty
        history = osc.get_history()
        assert len(history) == 1  # Should return current phases


class TestHierarchicalKuramotoSystem:
    """Tests for HierarchicalKuramotoSystem."""

    def test_get_phase_modulation_factor(self):
        """Should return phase modulation factor."""
        system = HierarchicalKuramotoSystem(n_levels=3)
        factor = system.get_phase_modulation_factor(level=0)
        # Should return cos(phi) for level 0
        assert -1.0 <= factor <= 1.0

    def test_get_phase_modulation_invalid_level(self):
        """Should return 0 for invalid level."""
        system = HierarchicalKuramotoSystem(n_levels=3)
        assert system.get_phase_modulation_factor(level=-1) == 0.0
        assert system.get_phase_modulation_factor(level=10) == 0.0
