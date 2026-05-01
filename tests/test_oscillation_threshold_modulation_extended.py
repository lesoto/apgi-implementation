"""Extended tests for oscillation/threshold_modulation.py to cover missing branches.

Covers lines 196, 203-212 in oscillation/threshold_modulation.py
"""

from __future__ import annotations

import numpy as np
import pytest

from oscillation.threshold_modulation import HierarchicalPhaseModulator, phase_amplitude_coupling


class TestHierarchicalPhaseModulatorExtended:
    """Extended tests for HierarchicalPhaseModulator to cover broadcast branches."""

    def test_reset_phase_with_broadcast(self) -> None:
        """Test phase reset with broadcast=True (covers lines 203-212)."""
        modulator = HierarchicalPhaseModulator(n_levels=4)

        # Set specific phases
        modulator.phases = np.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2])

        # Reset with broadcast
        modulator.reset_phase_on_ignition(level=1, reset_amount=np.pi, broadcast=True)

        # Level 1 should be reset by reset_amount
        expected_phase_1 = (np.pi / 2 + np.pi) % (2 * np.pi)
        assert modulator.phases[1] == pytest.approx(expected_phase_1)

        # Other levels should also be affected by broadcast
        # Level 0: distance=1, effective_reset = pi * (0.5^1) = pi/2
        expected_phase_0 = (0.0 + np.pi * 0.5) % (2 * np.pi)
        assert modulator.phases[0] == pytest.approx(expected_phase_0)

        # Level 2: distance=1, same as level 0
        expected_phase_2 = (np.pi + np.pi * 0.5) % (2 * np.pi)
        assert modulator.phases[2] == pytest.approx(expected_phase_2)

        # Level 3: distance=2, effective_reset = pi * (0.5^2) = pi/4
        expected_phase_3 = (3 * np.pi / 2 + np.pi * 0.25) % (2 * np.pi)
        assert modulator.phases[3] == pytest.approx(expected_phase_3)

    def test_reset_phase_broadcast_custom_decay(self) -> None:
        """Test phase reset with custom broadcast decay."""
        modulator = HierarchicalPhaseModulator(n_levels=3, broadcast_decay=0.3)

        modulator.phases = np.array([0.0, np.pi / 2, np.pi])

        # Reset level 1 with broadcast and custom decay
        modulator.reset_phase_on_ignition(level=1, reset_amount=np.pi, broadcast=True)

        # Level 0: distance=1, effective_reset = pi * (0.3^1) = 0.3*pi
        expected_phase_0 = (0.0 + np.pi * 0.3) % (2 * np.pi)
        assert modulator.phases[0] == pytest.approx(expected_phase_0)

    def test_reset_phase_broadcast_level_zero(self) -> None:
        """Test broadcast from level 0."""
        modulator = HierarchicalPhaseModulator(n_levels=3)

        modulator.phases = np.array([0.0, np.pi / 2, np.pi])

        # Reset level 0 with broadcast
        modulator.reset_phase_on_ignition(level=0, reset_amount=np.pi / 2, broadcast=True)

        # Level 0: direct reset
        expected_phase_0 = np.pi / 2  # (0 + pi/2) % 2pi
        assert modulator.phases[0] == pytest.approx(expected_phase_0)

        # Level 1: distance=1
        expected_phase_1 = (np.pi / 2 + np.pi / 2 * 0.5) % (2 * np.pi)
        assert modulator.phases[1] == pytest.approx(expected_phase_1)

    def test_reset_phase_broadcast_last_level(self) -> None:
        """Test broadcast from last level."""
        modulator = HierarchicalPhaseModulator(n_levels=3)

        modulator.phases = np.array([0.0, np.pi / 2, np.pi])

        # Reset level 2 (last) with broadcast
        modulator.reset_phase_on_ignition(level=2, reset_amount=np.pi / 2, broadcast=True)

        # Level 2: direct reset
        expected_phase_2 = (np.pi + np.pi / 2) % (2 * np.pi)
        assert modulator.phases[2] == pytest.approx(expected_phase_2)

        # Level 1: distance=1
        expected_phase_1 = (np.pi / 2 + np.pi / 2 * 0.5) % (2 * np.pi)
        assert modulator.phases[1] == pytest.approx(expected_phase_1)

    def test_reset_phase_invalid_level_no_broadcast(self) -> None:
        """Test reset with invalid level and no broadcast (line 196)."""
        modulator = HierarchicalPhaseModulator(n_levels=3)

        original_phases = np.array([0.0, np.pi / 2, np.pi])
        modulator.phases = original_phases.copy()

        # Invalid level (negative)
        modulator.reset_phase_on_ignition(level=-1, reset_amount=np.pi, broadcast=False)
        # Phases should remain unchanged
        np.testing.assert_array_almost_equal(modulator.phases, original_phases)

        # Invalid level (too high)
        modulator.reset_phase_on_ignition(level=5, reset_amount=np.pi, broadcast=False)
        # Phases should remain unchanged
        np.testing.assert_array_almost_equal(modulator.phases, original_phases)

    def test_reset_phase_invalid_level_with_broadcast(self) -> None:
        """Test reset with invalid level but broadcast=True (still returns early)."""
        modulator = HierarchicalPhaseModulator(n_levels=3)

        original_phases = np.array([0.0, np.pi / 2, np.pi])
        modulator.phases = original_phases.copy()

        # Invalid level with broadcast=True (should still return early at line 196)
        modulator.reset_phase_on_ignition(level=-1, reset_amount=np.pi, broadcast=True)
        # Phases should remain unchanged
        np.testing.assert_array_almost_equal(modulator.phases, original_phases)


class TestPhaseAmplitudeCouplingExtended:
    """Extended tests for phase_amplitude_coupling function."""

    def test_pac_with_different_amplitudes(self) -> None:
        """Test PAC with various amplitude values."""
        phase = np.pi / 4

        # Different amplitudes
        result_low = phase_amplitude_coupling(phase, 0.1)
        result_mid = phase_amplitude_coupling(phase, 1.0)
        result_high = phase_amplitude_coupling(phase, 2.0)

        # Results should scale with amplitude
        assert result_low < result_mid < result_high

    def test_pac_with_zero_amplitude(self) -> None:
        """Test PAC with zero amplitude."""
        phase = np.pi / 2
        result = phase_amplitude_coupling(phase, 0.0)
        assert result == 0.0

    def test_pac_negative_phase(self) -> None:
        """Test PAC with negative phase."""
        phase = -np.pi / 4
        amplitude = 1.0
        result = phase_amplitude_coupling(phase, amplitude)
        # cos(-pi/4) = cos(pi/4) > 0
        assert result > 0

    def test_pac_phase_wraparound(self) -> None:
        """Test PAC with phase > 2*pi."""
        phase = 5 * np.pi  # Wraps to pi
        amplitude = 1.0
        result = phase_amplitude_coupling(phase, amplitude)
        # cos(5*pi) = cos(pi) = -1, but we take max(0, ...)
        expected = max(0.0, np.cos(5 * np.pi)) * amplitude
        assert result == pytest.approx(expected)
