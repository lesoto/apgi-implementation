"""Unit tests for thermodynamics module (Landauer's principle).

Tests APGI Spec §11: Thermodynamic Constraints
"""

import numpy as np
import pytest

from core.thermodynamics import (
    compute_landauer_cost,
    compute_landauer_cost_batch,
    compute_information_bits,
    compute_metabolic_efficiency,
    estimate_temperature_from_cost,
    validate_thermodynamic_constraint,
    thermodynamic_cost_trajectory,
    K_BOLTZMANN,
    T_ENV_DEFAULT,
    LN2,
)


class TestLandauerCost:
    """Test Landauer cost computation."""

    def test_zero_signal(self):
        """Cost should be zero when S ≤ ε."""
        cost = compute_landauer_cost(S=0.0, eps=0.01)
        assert cost == 0.0

    def test_signal_equals_threshold(self):
        """Cost should be zero when S = ε."""
        cost = compute_landauer_cost(S=0.01, eps=0.01)
        assert cost == 0.0

    def test_signal_below_threshold(self):
        """Cost should be zero when S < ε."""
        cost = compute_landauer_cost(S=0.005, eps=0.01)
        assert cost == 0.0

    def test_positive_cost(self):
        """Cost should be positive when S > ε."""
        cost = compute_landauer_cost(S=1.0, eps=0.01)
        assert cost > 0.0

    def test_cost_increases_with_signal(self):
        """Cost should increase with signal magnitude."""
        cost_1 = compute_landauer_cost(S=1.0, eps=0.01)
        cost_2 = compute_landauer_cost(S=2.0, eps=0.01)
        assert cost_2 > cost_1

    def test_cost_logarithmic_scaling(self):
        """Cost should scale logarithmically with signal."""
        cost_1 = compute_landauer_cost(S=1.0, eps=0.01)
        cost_2 = compute_landauer_cost(S=2.0, eps=0.01)
        # log₂(2/0.01) / log₂(1/0.01) = log₂(200) / log₂(100) ≈ 1.69
        ratio = cost_2 / cost_1
        expected_ratio = np.log2(2.0 / 0.01) / np.log2(1.0 / 0.01)
        assert np.isclose(ratio, expected_ratio, rtol=1e-6)

    def test_cost_increases_with_temperature(self):
        """Cost should increase with environmental temperature."""
        cost_cold = compute_landauer_cost(S=1.0, eps=0.01, T_env=300.0)
        cost_hot = compute_landauer_cost(S=1.0, eps=0.01, T_env=320.0)
        assert cost_hot > cost_cold

    def test_cost_increases_with_kappa_meta(self):
        """Cost should increase with metabolic efficiency factor."""
        cost_1 = compute_landauer_cost(S=1.0, eps=0.01, kappa_meta=1.0)
        cost_2 = compute_landauer_cost(S=1.0, eps=0.01, kappa_meta=2.0)
        assert np.isclose(cost_2, 2.0 * cost_1)

    def test_invalid_eps(self):
        """Should raise ValueError for invalid eps."""
        with pytest.raises(ValueError):
            compute_landauer_cost(S=1.0, eps=0.0)
        with pytest.raises(ValueError):
            compute_landauer_cost(S=1.0, eps=-0.01)

    def test_invalid_k_b(self):
        """Should raise ValueError for invalid k_b."""
        with pytest.raises(ValueError):
            compute_landauer_cost(S=1.0, eps=0.01, k_b=0.0)
        with pytest.raises(ValueError):
            compute_landauer_cost(S=1.0, eps=0.01, k_b=-1e-23)

    def test_invalid_temperature(self):
        """Should raise ValueError for invalid temperature."""
        with pytest.raises(ValueError):
            compute_landauer_cost(S=1.0, eps=0.01, T_env=0.0)
        with pytest.raises(ValueError):
            compute_landauer_cost(S=1.0, eps=0.01, T_env=-310.0)

    def test_invalid_kappa_meta(self):
        """Should raise ValueError for invalid kappa_meta."""
        with pytest.raises(ValueError):
            compute_landauer_cost(S=1.0, eps=0.01, kappa_meta=0.0)
        with pytest.raises(ValueError):
            compute_landauer_cost(S=1.0, eps=0.01, kappa_meta=-1.0)

    def test_formula_correctness(self):
        """Test against known formula."""
        S = 1.0
        eps = 0.01
        k_b = K_BOLTZMANN
        T_env = T_ENV_DEFAULT
        kappa_meta = 1.0

        cost = compute_landauer_cost(S, eps, k_b, T_env, kappa_meta)

        # Manual computation
        n_erase = np.log2(S / eps)
        expected = kappa_meta * n_erase * k_b * T_env * LN2

        assert np.isclose(cost, expected, rtol=1e-10)


class TestBatchComputation:
    """Test batch Landauer cost computation."""

    def test_batch_matches_scalar(self):
        """Batch computation should match scalar computation."""
        S_vals = np.array([0.1, 0.5, 1.0, 2.0])
        costs_batch = compute_landauer_cost_batch(S_vals, eps=0.01)

        for i, S in enumerate(S_vals):
            cost_scalar = compute_landauer_cost(S, eps=0.01)
            assert np.isclose(costs_batch[i], cost_scalar)

    def test_batch_shape(self):
        """Batch output should have same shape as input."""
        S_vals = np.array([0.1, 0.5, 1.0, 2.0])
        costs = compute_landauer_cost_batch(S_vals, eps=0.01)
        assert costs.shape == S_vals.shape

    def test_batch_with_zeros(self):
        """Batch should handle zero signals correctly."""
        S_vals = np.array([0.0, 0.5, 1.0])
        costs = compute_landauer_cost_batch(S_vals, eps=0.01)
        assert costs[0] == 0.0
        assert costs[1] > 0.0
        assert costs[2] > 0.0


class TestInformationBits:
    """Test information bit computation."""

    def test_zero_signal(self):
        """Bits should be zero when S ≤ ε."""
        bits = compute_information_bits(S=0.0, eps=0.01)
        assert bits == 0.0

    def test_positive_bits(self):
        """Bits should be positive when S > ε."""
        bits = compute_information_bits(S=1.0, eps=0.01)
        assert bits > 0.0

    def test_formula_correctness(self):
        """Test against known formula."""
        S = 1.0
        eps = 0.01
        bits = compute_information_bits(S, eps)
        expected = np.log2(S / eps)
        assert np.isclose(bits, expected)


class TestMetabolicEfficiency:
    """Test metabolic efficiency factor computation."""

    def test_efficiency_from_cost(self):
        """Should recover kappa_meta from cost."""
        S = 1.0
        eps = 0.01
        kappa_meta_true = 2.0

        cost = compute_landauer_cost(S, eps, kappa_meta=kappa_meta_true)
        kappa_meta_est = compute_metabolic_efficiency(cost, S, eps)

        assert np.isclose(kappa_meta_est, kappa_meta_true, rtol=1e-6)

    def test_invalid_signal(self):
        """Should raise ValueError when S ≤ eps."""
        with pytest.raises(ValueError):
            compute_metabolic_efficiency(C_metabolic=1e-20, S=0.005, eps=0.01)

    def test_zero_denominator_efficiency(self):
        """Should raise ValueError when denominator is zero."""
        with pytest.raises(ValueError, match="Denominator is zero"):
            compute_metabolic_efficiency(C_metabolic=1e-20, S=1.0, eps=0.01, k_b=0.0)


class TestTemperatureEstimation:
    """Test temperature estimation from cost."""

    def test_temperature_from_cost(self):
        """Should recover T_env from cost."""
        S = 1.0
        eps = 0.01
        T_env_true = 310.0

        cost = compute_landauer_cost(S, eps, T_env=T_env_true)
        T_env_est = estimate_temperature_from_cost(cost, S, eps)

        assert np.isclose(T_env_est, T_env_true, rtol=1e-6)

    def test_invalid_signal(self):
        """Should raise ValueError when S ≤ eps."""
        with pytest.raises(ValueError):
            estimate_temperature_from_cost(C_metabolic=1e-20, S=0.005, eps=0.01)

    def test_zero_denominator_estimate_temperature(self):
        """Should raise ValueError when denominator is zero."""
        with pytest.raises(ValueError, match="Denominator is zero"):
            estimate_temperature_from_cost(
                C_metabolic=1e-20, S=1.0, eps=0.01, kappa_meta=0.0
            )


class TestThermodynamicConstraintValidation:
    """Test thermodynamic constraint validation."""

    def test_constraint_satisfied(self):
        """Should pass when C ≥ E_min."""
        S = 1.0
        eps = 0.01
        E_min = compute_landauer_cost(S, eps)
        C = 2.0 * E_min  # Twice the minimum

        result = validate_thermodynamic_constraint(C, S, eps)

        assert result["satisfied"] is True
        assert result["ratio"] >= 1.0

    def test_constraint_violated(self):
        """Should fail when C < E_min."""
        S = 1.0
        eps = 0.01
        E_min = compute_landauer_cost(S, eps)
        C = 0.5 * E_min  # Half the minimum

        result = validate_thermodynamic_constraint(C, S, eps, tolerance=0.01)

        assert result["satisfied"] is False
        assert result["violation"] > 0.0

    def test_zero_signal(self):
        """Should pass when S ≤ ε (no information)."""
        result = validate_thermodynamic_constraint(C_metabolic=0.0, S=0.005, eps=0.01)
        assert result["satisfied"] is True


class TestTrajectoryAnalysis:
    """Test thermodynamic cost trajectory analysis."""

    def test_trajectory_shape(self):
        """Trajectory analysis should return correct shapes."""
        S_hist = np.array([0.1, 0.5, 1.0, 0.5, 0.1])
        result = thermodynamic_cost_trajectory(S_hist, eps=0.01)

        assert result["costs"].shape == S_hist.shape
        assert result["bits_history"].shape == S_hist.shape
        assert len(result["costs"]) == len(S_hist)

    def test_trajectory_statistics(self):
        """Trajectory statistics should be computed correctly."""
        S_hist = np.array([0.1, 0.5, 1.0, 0.5, 0.1])
        result = thermodynamic_cost_trajectory(S_hist, eps=0.01)

        assert result["total_cost"] == np.sum(result["costs"])
        assert result["mean_cost"] == np.mean(result["costs"])
        assert result["max_cost"] == np.max(result["costs"])
        assert result["min_cost"] == np.min(result["costs"])

    def test_trajectory_monotonicity(self):
        """Cost should increase with signal magnitude."""
        S_hist = np.array([0.1, 0.5, 1.0, 2.0])
        result = thermodynamic_cost_trajectory(S_hist, eps=0.01)

        # Costs should be non-decreasing (allowing for numerical precision)
        for i in range(len(result["costs"]) - 1):
            assert result["costs"][i + 1] >= result["costs"][i] - 1e-10


class TestPhysicalConstants:
    """Test physical constants."""

    def test_boltzmann_constant(self):
        """Boltzmann constant should be correct."""
        assert np.isclose(K_BOLTZMANN, 1.38e-23, rtol=1e-2)

    def test_ln2(self):
        """ln(2) should be correct."""
        assert np.isclose(LN2, np.log(2.0), rtol=1e-10)

    def test_default_temperature(self):
        """Default temperature should be body temperature."""
        assert T_ENV_DEFAULT == 310.0  # 37°C in Kelvin


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
