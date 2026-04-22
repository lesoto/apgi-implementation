"""Extended tests for core/thermodynamics.py to achieve 100% coverage."""

import numpy as np
import pytest
from core.thermodynamics import (
    compute_landauer_cost,
    compute_landauer_cost_batch,
    compute_metabolic_efficiency,
    estimate_temperature_from_cost,
    validate_thermodynamic_constraint,
)


class TestComputeLandauerCostExtended:
    """Extended tests for compute_landauer_cost."""

    def test_edge_case_equal_to_eps(self):
        """Should return 0 when S equals eps."""
        result = compute_landauer_cost(S=0.01, eps=0.01)
        assert result == 0.0


class TestComputeLandauerCostBatchExtended:
    """Extended tests for compute_landauer_cost_batch."""

    def test_batch_with_mixed_values(self):
        """Should handle array with values above and below eps."""
        S_array = np.array([0.001, 0.01, 0.1, 1.0])  # 0.001 and 0.01 are <= eps
        eps = 0.01
        result = compute_landauer_cost_batch(S_array, eps)
        assert len(result) == len(S_array)
        # First two should be 0 (S <= eps)
        assert result[0] == 0.0
        assert result[1] == 0.0
        # Last two should be positive
        assert result[2] > 0
        assert result[3] > 0


class TestValidateThermodynamicConstraint:
    """Extended tests for validate_thermodynamic_constraint."""

    def test_zero_e_min_case(self):
        """Should handle case where E_min = 0."""
        result = validate_thermodynamic_constraint(
            C_metabolic=1.0,
            S=0.001,  # S <= eps, so E_min = 0
            eps=0.01,
        )
        assert result["satisfied"] is True
        assert result["E_min"] == 0.0


class TestComputeMetabolicEfficiency:
    """Extended tests for compute_metabolic_efficiency."""

    def test_s_le_eps_error(self):
        """Should raise ValueError when S <= eps."""
        with pytest.raises(ValueError, match="S must be > eps"):
            compute_metabolic_efficiency(
                C_metabolic=1.0,
                S=0.5,  # S <= eps
                eps=1.0,
            )


class TestEstimateTemperatureFromCost:
    """Extended tests for estimate_temperature_from_cost."""

    def test_s_le_eps_error(self):
        """Should raise ValueError when S <= eps."""
        with pytest.raises(ValueError, match="S must be > eps"):
            estimate_temperature_from_cost(
                C_metabolic=1.0,
                S=0.5,  # S <= eps
                eps=1.0,
            )
