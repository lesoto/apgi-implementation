"""Comprehensive unit tests for oscillation/kuramoto.py module.

Tests cover:
- OrnsteinUhlenbeckNoise class
- KuramotoOscillators class
- HierarchicalKuramotoSystem class
"""

from __future__ import annotations

import numpy as np
import pytest

from oscillation.kuramoto import (
    HierarchicalKuramotoSystem,
    KuramotoOscillators,
    OrnsteinUhlenbeckNoise,
)


class TestOrnsteinUhlenbeckNoise:
    """Tests for OrnsteinUhlenbeckNoise class."""

    def test_initialization(self):
        """Should initialize with correct parameters."""
        noise = OrnsteinUhlenbeckNoise(tau_xi=2.0, sigma_xi=0.2)
        assert noise.tau_xi == 2.0
        assert noise.sigma_xi == 0.2
        assert noise.xi == 0.0

    def test_step(self):
        """Should update noise state."""
        noise = OrnsteinUhlenbeckNoise(tau_xi=1.0, sigma_xi=0.1)
        initial_xi = noise.xi

        result = noise.step(dt=1.0)

        # Noise should have changed
        assert result != initial_xi or noise.xi != initial_xi

    def test_reverts_to_zero(self):
        """Should revert toward zero over time."""
        noise = OrnsteinUhlenbeckNoise(tau_xi=1.0, sigma_xi=0.1)
        noise.xi = 10.0  # Set far from zero

        # Step multiple times
        for _ in range(100):
            noise.step(dt=0.1)

        # Should be closer to zero
        assert abs(noise.xi) < 10.0

    def test_reset(self):
        """Should reset to zero."""
        noise = OrnsteinUhlenbeckNoise(tau_xi=1.0, sigma_xi=0.1)
        noise.step(dt=1.0)
        noise.reset()
        assert noise.xi == 0.0


class TestKuramotoOscillators:
    """Tests for KuramotoOscillators class."""

    def test_initialization(self):
        """Should initialize with correct parameters."""
        osc = KuramotoOscillators(n_levels=3)
        assert osc.n_levels == 3
        assert len(osc.phases) == 3
        assert len(osc.omegas) == 3
        assert len(osc.noise_processes) == 3

    def test_custom_frequencies(self):
        """Should accept custom frequencies."""
        freqs = np.array([1.0, 5.0, 10.0])
        osc = KuramotoOscillators(n_levels=3, frequencies=freqs)

        # Converted to rad/ms
        expected = 2 * np.pi * freqs / 1000.0
        np.testing.assert_array_equal(osc.omegas, expected)

    def test_step(self):
        """Should update phases."""
        osc = KuramotoOscillators(n_levels=3)
        initial_phases = osc.phases.copy()

        new_phases = osc.step(dt=1.0)

        # Phases should have changed
        assert not np.allclose(new_phases, initial_phases)

    def test_get_phases(self):
        """Should return phases."""
        osc = KuramotoOscillators(n_levels=3)
        phases = osc.get_phases()
        assert len(phases) == 3

    def test_set_phases(self):
        """Should set phases correctly."""
        osc = KuramotoOscillators(n_levels=3)
        osc.set_phases(np.array([0.5, 1.0, 1.5]))

        np.testing.assert_array_equal(osc.phases, np.array([0.5, 1.0, 1.5]))

    def test_reset_phase_on_ignition(self):
        """Should reset phase on ignition."""
        osc = KuramotoOscillators(n_levels=3)
        initial_phase = osc.phases[1].copy()

        osc.reset_phase_on_ignition(level=1, reset_amount=np.pi)

        # Phase should have changed by approximately pi
        phase_diff = abs(osc.phases[1] - initial_phase)
        # Account for wraparound
        phase_diff = min(phase_diff, 2 * np.pi - phase_diff)
        assert pytest.approx(phase_diff, abs=0.1) == np.pi

    def test_reset_phase_on_ignition_out_of_bounds(self):
        """Should return early if level is out of bounds."""
        osc = KuramotoOscillators(n_levels=3)
        initial_phases = osc.phases.copy()

        osc.reset_phase_on_ignition(level=-1)
        osc.reset_phase_on_ignition(level=3)

        np.testing.assert_array_equal(osc.phases, initial_phases)

    def test_get_synchronization_order(self):
        """Should compute synchronization order."""
        osc = KuramotoOscillators(n_levels=3)

        # Set all phases to same value for perfect synchronization
        osc.set_phases(np.array([0.0, 0.0, 0.0]))
        sync = osc.get_synchronization_order()

        # Perfect synchronization should give R ≈ 1
        assert pytest.approx(sync, abs=0.01) == 1.0

    def test_get_phase_coherence(self):
        """Should compute phase coherence."""
        osc = KuramotoOscillators(n_levels=3)
        coherence = osc.get_phase_coherence()

        assert coherence.shape == (3, 3)
        # Diagonal should be 1 (coherence with self)
        np.testing.assert_array_almost_equal(np.diag(coherence), np.ones(3))

    def test_get_history(self):
        """Should return phase history."""
        osc = KuramotoOscillators(n_levels=3)

        # Step multiple times
        for _ in range(5):
            osc.step(dt=1.0)

        history = osc.get_history()
        assert history.shape[0] == 5  # 5 time steps
        assert history.shape[1] == 3  # 3 oscillators


class TestHierarchicalKuramotoSystem:
    """Tests for HierarchicalKuramotoSystem class."""

    def test_initialization(self):
        """Should initialize correctly."""
        system = HierarchicalKuramotoSystem(n_levels=3)
        assert system.n_levels == 3
        assert system.oscillators is not None

    def test_initialization_with_config(self):
        """Should accept configuration."""
        config = {"kuramoto_tau_xi": 2.0, "kuramoto_sigma_xi": 0.2}
        system = HierarchicalKuramotoSystem(n_levels=3, config=config)
        assert system.n_levels == 3

    def test_step(self):
        """Should update and return diagnostics."""
        system = HierarchicalKuramotoSystem(n_levels=3)
        result = system.step(dt=1.0)

        assert "phases" in result
        assert "synchronization" in result
        assert "coherence" in result
        assert len(result["phases"]) == 3

    def test_apply_ignition_reset(self):
        """Should apply phase reset on ignition."""
        system = HierarchicalKuramotoSystem(n_levels=3)
        initial_phase = system.oscillators.phases[1].copy()

        system.apply_ignition_reset(level=1)

        # Phase should have changed
        assert system.oscillators.phases[1] != initial_phase

    def test_get_phase_modulation_factor(self):
        """Should return phase modulation factor."""
        system = HierarchicalKuramotoSystem(n_levels=3)

        # Set known phase
        system.oscillators.set_phases(np.array([0.0, 0.0, 0.0]))
        factor = system.get_phase_modulation_factor(level=0)

        # cos(0) = 1
        assert pytest.approx(factor, abs=0.01) == 1.0

    def test_invalid_level(self):
        """Should return 0 for invalid level."""
        system = HierarchicalKuramotoSystem(n_levels=3)
        factor = system.get_phase_modulation_factor(level=10)
        assert factor == 0.0
