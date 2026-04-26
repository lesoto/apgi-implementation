"""Comprehensive unit tests for energy/thermodynamics.py module.

Tests cover:
- estimate_bits_erased function
- metabolic_cost_landauer function
- metabolic_cost function
- landauer_limit function
- landauer_cost_in_atp function
- estimate_information_content function
- check_thermodynamic_feasibility function
- ThermodynamicTracker class
"""

from __future__ import annotations

import numpy as np
import pytest

from energy.thermodynamics import (
    ThermodynamicTracker,
    check_thermodynamic_feasibility,
    estimate_bits_erased,
    estimate_information_content,
    landauer_cost_in_atp,
    landauer_limit,
    metabolic_cost,
    metabolic_cost_landauer,
)


class TestEstimateBitsErased:
    """Tests for estimate_bits_erased function."""

    def test_basic_estimation(self):
        """Should estimate bits correctly."""
        result = estimate_bits_erased(S=2.0, eps_stab=1e-6)
        # N = log2(2.0 / 1e-6) = log2(2e6)
        expected = np.log2(2e6)
        assert pytest.approx(result, rel=1e-7) == expected

    def test_zero_signal(self):
        """Should return 0 for zero signal."""
        result = estimate_bits_erased(S=0.0, eps_stab=1e-6)
        assert result == 0.0

    def test_negative_signal(self):
        """Should return 0 for negative signal."""
        result = estimate_bits_erased(S=-1.0, eps_stab=1e-6)
        assert result == 0.0


class TestMetabolicCostLandauer:
    """Tests for metabolic_cost_landauer function."""

    def test_basic_computation(self):
        """Should compute Landauer cost correctly."""
        result = metabolic_cost_landauer(
            N_erase=10.0,
            kappa_meta=1.0,
            T_env=310.0,
        )
        # C = 1.0 * 10.0 * 1.38e-23 * 310 * ln(2)
        expected = 10.0 * 1.380649e-23 * 310.0 * np.log(2)
        assert pytest.approx(result, rel=1e-7) == expected

    def test_zero_bits(self):
        """Should return 0 for zero bits."""
        result = metabolic_cost_landauer(
            N_erase=0.0,
            kappa_meta=1.0,
            T_env=310.0,
        )
        assert result == 0.0


class TestMetabolicCost:
    """Tests for metabolic_cost function."""

    def test_basic_computation(self):
        """Should compute cost in ATP."""
        result = metabolic_cost(kappa=100.0, bits=10.0)
        assert result == 1000.0

    def test_zero_bits(self):
        """Should return 0 for zero bits."""
        result = metabolic_cost(kappa=100.0, bits=0.0)
        assert result == 0.0


class TestLandauerLimit:
    """Tests for landauer_limit function."""

    def test_default_temperature(self):
        """Should use body temperature by default."""
        result = landauer_limit()
        expected = 1.380649e-23 * 310.0 * np.log(2)
        assert pytest.approx(result, rel=1e-7) == expected

    def test_custom_temperature(self):
        """Should accept custom temperature."""
        result = landauer_limit(T=300.0)
        expected = 1.380649e-23 * 300.0 * np.log(2)
        assert pytest.approx(result, rel=1e-7) == expected


class TestLandauerCostInATP:
    """Tests for landauer_cost_in_atp function."""

    def test_basic_computation(self):
        """Should convert to ATP equivalents."""
        result = landauer_cost_in_atp(bits=10.0, T=310.0)
        # Energy in joules / energy per ATP
        energy_joules = 1.380649e-23 * 310.0 * np.log(2) * 10.0
        expected = energy_joules / 5.2e-21
        assert pytest.approx(result, rel=1e-7) == expected


class TestEstimateInformationContent:
    """Tests for estimate_information_content function."""

    def test_basic_computation(self):
        """Should estimate information content."""
        result = estimate_information_content(
            z_e=1.0,
            z_i=0.5,
            pi_e=2.0,
            pi_i=1.0,
            bits_per_unit=0.1,
        )
        # info = (2.0*1.0 + 1.0*0.25) * 0.1
        assert result > 0

    def test_zero_errors(self):
        """Should return 0 for zero errors."""
        result = estimate_information_content(
            z_e=0.0,
            z_i=0.0,
            pi_e=2.0,
            pi_i=1.0,
        )
        assert result == 0.0


class TestCheckThermodynamicFeasibility:
    """Tests for check_thermodynamic_feasibility function."""

    def test_feasible_cost(self):
        """Should return feasible for high ATP cost."""
        result = check_thermodynamic_feasibility(
            bits=10.0,
            atp_cost=10000.0,
            efficiency=0.1,
            T=310.0,
        )
        assert result["is_feasible"] is True

    def test_infeasible_cost(self):
        """Should return infeasible for low ATP cost."""
        result = check_thermodynamic_feasibility(
            bits=1000.0,
            atp_cost=1.0,
            efficiency=0.1,
            T=310.0,
        )
        assert result["is_feasible"] is False

    def test_landauer_violation(self):
        """Should detect Landauer violation."""
        # Cost below theoretical minimum
        result = check_thermodynamic_feasibility(
            bits=100.0,
            atp_cost=1e-10,
            efficiency=1.0,
            T=310.0,
        )
        assert result["landauer_violation"] is True


class TestThermodynamicTracker:
    """Tests for ThermodynamicTracker class."""

    def test_initialization(self):
        """Should initialize correctly."""
        tracker = ThermodynamicTracker(kappa=100.0, efficiency=0.1, temperature=310.0)
        assert tracker.kappa == 100.0
        assert tracker.efficiency == 0.1
        assert tracker.temperature == 310.0

    def test_record_ignition(self):
        """Should record ignition event."""
        tracker = ThermodynamicTracker()
        result = tracker.record_ignition(
            z_e=1.0,
            z_i=0.5,
            pi_e=2.0,
            pi_i=1.0,
        )
        assert "bits" in result
        assert "atp_cost" in result
        assert "feasibility" in result

    def test_get_summary(self):
        """Should return summary of recordings."""
        tracker = ThermodynamicTracker()

        # Record multiple ignitions
        for _ in range(5):
            tracker.record_ignition(z_e=1.0, z_i=0.5, pi_e=2.0, pi_i=1.0)

        summary = tracker.get_summary()
        assert summary["total_ignitions"] == 5
        assert summary["total_bits_processed"] > 0
        assert summary["total_atp_cost"] > 0

    def test_validate_total(self):
        """Should validate total costs."""
        tracker = ThermodynamicTracker()

        # Record multiple ignitions
        for _ in range(10):
            tracker.record_ignition(z_e=1.0, z_i=0.5, pi_e=2.0, pi_i=1.0)

        validation = tracker.validate_total()
        assert "is_physically_possible" in validation
        assert "is_biologically_plausible" in validation
