"""Comprehensive unit tests for oscillation/threshold_modulation.py module.

Tests cover:
- compute_modulation_factor function
- modulate_threshold_by_phase function
- hierarchical_threshold_modulation function
- compute_phase_window function
- phase_gated_ignition_probability function
"""

from __future__ import annotations

import numpy as np
import pytest

from oscillation.threshold_modulation import (
    compute_modulation_factor,
    modulate_threshold_by_phase,
    hierarchical_threshold_modulation,
    compute_phase_window,
    phase_gated_ignition_probability,
)


class TestComputeModulationFactor:
    """Tests for compute_modulation_factor function."""

    def test_cosine_modulation(self):
        """Should compute cosine modulation."""
        result = compute_modulation_factor(
            pi_higher=1.0,
            phi_higher=0.0,
            kappa=0.1,
            modulation_type="cosine",
        )
        # 1 + 0.1 * 1.0 * 1.0 = 1.1
        assert pytest.approx(result, rel=1e-7) == 1.1

    def test_sine_modulation(self):
        """Should compute sine modulation."""
        result = compute_modulation_factor(
            pi_higher=1.0,
            phi_higher=np.pi / 2,
            kappa=0.1,
            modulation_type="sine",
        )
        # 1 + 0.1 * 1.0 * 1.0 = 1.1
        assert pytest.approx(result, rel=1e-7) == 1.1

    def test_rectified_modulation(self):
        """Should compute rectified modulation."""
        result = compute_modulation_factor(
            pi_higher=1.0,
            phi_higher=np.pi,
            kappa=0.1,
            modulation_type="rectified",
        )
        # max(0, cos(pi)) = 0, so modulation = 1.0
        assert result == 1.0

    def test_invalid_modulation_type(self):
        """Should raise ValueError for invalid type."""
        with pytest.raises(ValueError, match="Unknown modulation_type"):
            compute_modulation_factor(
                pi_higher=1.0,
                phi_higher=0.0,
                kappa=0.1,
                modulation_type="invalid",
            )


class TestModulateThresholdByPhase:
    """Tests for modulate_threshold_by_phase function."""

    def test_increase_threshold(self):
        """Should increase threshold at preferred phase."""
        result = modulate_threshold_by_phase(
            theta_baseline=1.0,
            pi_higher=1.0,
            phi_higher=0.0,
            kappa=0.1,
        )
        assert result > 1.0

    def test_decrease_threshold(self):
        """Should decrease threshold at opposite phase."""
        result = modulate_threshold_by_phase(
            theta_baseline=1.0,
            pi_higher=1.0,
            phi_higher=np.pi,
            kappa=0.1,
        )
        assert result < 1.0

    def test_clamping_min(self):
        """Should clamp to minimum."""
        result = modulate_threshold_by_phase(
            theta_baseline=0.05,
            pi_higher=10.0,
            phi_higher=np.pi,
            kappa=0.5,
            theta_min=0.1,
        )
        assert result == 0.1

    def test_clamping_max(self):
        """Should clamp to maximum."""
        result = modulate_threshold_by_phase(
            theta_baseline=50.0,
            pi_higher=10.0,
            phi_higher=0.0,
            kappa=0.5,
            theta_max=100.0,
        )
        assert result <= 100.0


class TestHierarchicalThresholdModulation:
    """Tests for hierarchical_threshold_modulation function."""

    def test_basic_modulation(self):
        """Should modulate all non-top levels."""
        thetas = np.array([1.0, 1.0, 1.0])
        pis = np.array([1.0, 1.0, 1.0])
        phases = np.array([0.0, np.pi / 2, np.pi])

        result = hierarchical_threshold_modulation(
            thetas=thetas,
            pis=pis,
            phases=phases,
            kappa_down=0.1,
        )

        assert len(result) == 3
        # Top level should be unchanged
        assert result[2] == 1.0
        # Lower levels should be modulated
        assert result[0] != 1.0 or result[1] != 1.0


class TestComputePhaseWindow:
    """Tests for compute_phase_window function."""

    def test_center_of_window(self):
        """Should return 1 at center of window."""
        result = compute_phase_window(
            phi=0.0,
            window_center=0.0,
            window_width=np.pi / 2,
        )
        assert result == 1.0

    def test_outside_window(self):
        """Should return 0 outside window."""
        result = compute_phase_window(
            phi=np.pi,
            window_center=0.0,
            window_width=np.pi / 2,
        )
        assert result == 0.0

    def test_falloff_inside_window(self):
        """Should fall off inside window."""
        result_center = compute_phase_window(
            phi=0.0,
            window_center=0.0,
            window_width=np.pi,
        )
        result_edge = compute_phase_window(
            phi=np.pi / 4,
            window_center=0.0,
            window_width=np.pi,
        )
        assert result_center > result_edge


class TestPhaseGatedIgnitionProbability:
    """Tests for phase_gated_ignition_probability function."""

    def test_center_boost(self):
        """Should boost probability at center of window."""
        result = phase_gated_ignition_probability(
            p_base=0.5,
            phi=0.0,
            window_center=0.0,
            window_width=np.pi / 2,
        )
        # p = 0.5 * (0.5 + 0.5 * 1.0) = 0.5
        assert result == 0.5

    def test_outside_reduction(self):
        """Should reduce probability outside window."""
        result = phase_gated_ignition_probability(
            p_base=0.5,
            phi=np.pi,
            window_center=0.0,
            window_width=np.pi / 2,
        )
        # p = 0.5 * (0.5 + 0.5 * 0.0) = 0.25
        assert result == 0.25

    def test_window_factor_scaling(self):
        """Should scale probability by window factor."""
        result = phase_gated_ignition_probability(
            p_base=0.8,
            phi=0.0,
            window_center=0.0,
            window_width=np.pi / 2,
        )
        # window_factor = 1.0 at center
        # p = 0.8 * (0.5 + 0.5 * 1.0) = 0.8
        assert result == 0.8
