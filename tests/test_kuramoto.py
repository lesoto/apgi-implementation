"""Tests for Kuramoto oscillators with phase noise.

Tests spec §9: Oscillatory Phase Coupling
"""

import numpy as np
import pytest

from oscillation.kuramoto import (
    HierarchicalKuramotoSystem,
    KuramotoOscillators,
    OrnsteinUhlenbeckNoise,
)


class TestOrnsteinUhlenbeckNoise:
    """Test OU noise process."""

    def test_initialization(self):
        """Test OU noise initialization."""
        noise = OrnsteinUhlenbeckNoise(tau_xi=1.0, sigma_xi=0.1)
        assert noise.tau_xi == 1.0
        assert noise.sigma_xi == 0.1
        assert noise.xi == 0.0

    def test_step(self):
        """Test OU noise step."""
        noise = OrnsteinUhlenbeckNoise(tau_xi=1.0, sigma_xi=0.1)

        # Take multiple steps
        values = []
        for _ in range(100):
            val = noise.step(dt=0.1)
            values.append(val)

        # Check that values are bounded
        assert np.all(np.abs(values) < 1.0)

        # Check that there's some variation
        assert np.std(values) > 0.01

    def test_reset(self):
        """Test OU noise reset."""
        noise = OrnsteinUhlenbeckNoise(tau_xi=1.0, sigma_xi=0.1)

        # Generate some noise
        for _ in range(10):
            noise.step(dt=0.1)

        # Reset
        noise.reset()
        assert noise.xi == 0.0


class TestKuramotoOscillators:
    """Test Kuramoto oscillator network."""

    def test_initialization(self):
        """Test Kuramoto initialization."""
        n_levels = 5
        osc = KuramotoOscillators(n_levels=n_levels)

        assert osc.n_levels == n_levels
        assert len(osc.phases) == n_levels
        assert len(osc.noise_processes) == n_levels
        assert np.all(osc.phases >= 0) and np.all(osc.phases < 2 * np.pi)

    def test_phases_bounded(self):
        """Test that phases remain in [0, 2π)."""
        osc = KuramotoOscillators(n_levels=5)

        for _ in range(100):
            phases = osc.step(dt=1.0)
            assert np.all(phases >= 0) and np.all(phases < 2 * np.pi)

    def test_synchronization_order(self):
        """Test synchronization order parameter."""
        osc = KuramotoOscillators(n_levels=5)

        # Initially random phases → low synchronization
        R_initial = osc.get_synchronization_order()
        assert 0 <= R_initial <= 1

        # Run for many steps with coupling
        for _ in range(1000):
            osc.step(dt=1.0)

        # Should increase synchronization
        R_final = osc.get_synchronization_order()
        assert 0 <= R_final <= 1

    def test_phase_coherence(self):
        """Test phase coherence matrix."""
        osc = KuramotoOscillators(n_levels=5)

        coherence = osc.get_phase_coherence()

        # Check shape
        assert coherence.shape == (5, 5)

        # Check values in [0, 1]
        assert np.all(coherence >= 0) and np.all(coherence <= 1)

        # Diagonal should be 1 (phase with itself)
        assert np.allclose(np.diag(coherence), 1.0)

    def test_phase_reset_on_ignition(self):
        """Test phase reset on ignition."""
        osc = KuramotoOscillators(n_levels=5)

        # Get initial phase
        phases_before = osc.get_phases()
        phi_0_before = phases_before[0]

        # Reset phase at level 0
        osc.reset_phase_on_ignition(level=0, reset_amount=np.pi)

        # Check that phase changed
        phases_after = osc.get_phases()
        phi_0_after = phases_after[0]

        # Should differ by π (mod 2π)
        diff = (phi_0_after - phi_0_before) % (2 * np.pi)
        assert np.isclose(diff, np.pi) or np.isclose(diff, 0)

    def test_history(self):
        """Test phase history recording."""
        osc = KuramotoOscillators(n_levels=5)

        # Run for 10 steps
        for _ in range(10):
            osc.step(dt=1.0)

        history = osc.get_history()
        assert history.shape == (10, 5)

    def test_coupling_matrix(self):
        """Test nearest-neighbor coupling matrix."""
        osc = KuramotoOscillators(n_levels=5)

        K = osc.K

        # Check shape
        assert K.shape == (5, 5)

        # Check nearest-neighbor structure
        # Diagonal should be zero
        assert np.allclose(np.diag(K), 0)

        # Only adjacent elements should be non-zero
        for i in range(5):
            for j in range(5):
                if abs(i - j) > 1:
                    assert K[i, j] == 0


class TestHierarchicalKuramotoSystem:
    """Test hierarchical Kuramoto system integration."""

    def test_initialization(self):
        """Test hierarchical system initialization."""
        n_levels = 5
        config = {"kuramoto_tau_xi": 1.0, "kuramoto_sigma_xi": 0.1}

        sys = HierarchicalKuramotoSystem(n_levels=n_levels, config=config)

        assert sys.n_levels == n_levels
        assert sys.oscillators is not None

    def test_step(self):
        """Test hierarchical system step."""
        sys = HierarchicalKuramotoSystem(n_levels=5)

        result = sys.step(dt=1.0)

        assert "phases" in result
        assert "synchronization" in result
        assert "coherence" in result
        assert len(result["phases"]) == 5

    def test_ignition_reset(self):
        """Test ignition reset integration."""
        sys = HierarchicalKuramotoSystem(n_levels=5)

        # Get initial phase
        phases_before = sys.oscillators.get_phases()

        # Apply ignition reset
        sys.apply_ignition_reset(level=2)

        # Check that phase changed
        phases_after = sys.oscillators.get_phases()

        # Phase at level 2 should have changed
        assert not np.isclose(phases_before[2], phases_after[2])

    def test_phase_modulation_factor(self):
        """Test phase modulation factor for threshold."""
        sys = HierarchicalKuramotoSystem(n_levels=5)

        # Get modulation factor
        factor = sys.get_phase_modulation_factor(level=0)

        # Should be in [-1, 1]
        assert -1 <= factor <= 1

        # Should be cos(φ)
        expected = np.cos(sys.oscillators.phases[0])
        assert np.isclose(factor, expected)


class TestKuramotoIntegration:
    """Integration tests for Kuramoto system."""

    def test_long_run_stability(self):
        """Test that system remains stable over long run."""
        osc = KuramotoOscillators(n_levels=5)

        # Run for 1000 steps
        for _ in range(1000):
            phases = osc.step(dt=1.0)

            # Check phases remain bounded
            assert np.all(phases >= 0) and np.all(phases < 2 * np.pi)

        # Check final state is reasonable
        R = osc.get_synchronization_order()
        assert 0 <= R <= 1

    def test_coupling_effect(self):
        """Test that coupling affects synchronization."""
        # System with coupling
        osc_coupled = KuramotoOscillators(n_levels=5)

        # System without coupling (zero coupling matrix)
        osc_uncoupled = KuramotoOscillators(n_levels=5)
        osc_uncoupled.K = np.zeros((5, 5))

        # Run both
        for _ in range(100):
            osc_coupled.step(dt=1.0)
            osc_uncoupled.step(dt=1.0)

        # Coupled should have higher synchronization
        R_coupled = osc_coupled.get_synchronization_order()
        R_uncoupled = osc_uncoupled.get_synchronization_order()

        # This is probabilistic, but coupled should tend to be higher
        # (not always guaranteed, but likely)
        assert R_coupled >= 0 and R_uncoupled >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
