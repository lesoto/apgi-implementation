"""Comprehensive unit tests for analysis/stability.py module.

Tests cover:
- compute_jacobian_discrete function
- compute_eigenvalues function
- check_stability function
- compute_fixed_point function
- analyze_bifurcation function
- validate_system_dynamics function
- StabilityAnalyzer class
"""

from __future__ import annotations

import numpy as np
import pytest

from analysis.stability import (
    compute_jacobian_discrete,
    compute_eigenvalues,
    check_stability,
    compute_fixed_point,
    analyze_bifurcation,
    validate_system_dynamics,
    StabilityAnalyzer,
)


class TestComputeJacobianDiscrete:
    """Tests for compute_jacobian_discrete function."""

    def test_basic_computation(self):
        """Should compute Jacobian correctly."""
        result = compute_jacobian_discrete(
            lam=0.2,
            kappa=0.15,
            c1=0.2,
            eta=0.1,
        )

        expected = np.array(
            [
                [1.0 - 0.2, 0.0],
                [0.1 * 0.2 * 0.2, np.exp(-0.15)],
            ]
        )

        np.testing.assert_array_almost_equal(result, expected)

    def test_diagonal_values(self):
        """Should have correct diagonal values."""
        result = compute_jacobian_discrete(
            lam=0.3,
            kappa=0.2,
            c1=0.2,
            eta=0.1,
        )

        assert result[0, 0] == 0.7  # 1 - 0.3
        assert pytest.approx(result[1, 1], rel=1e-7) == np.exp(-0.2)


class TestComputeEigenvalues:
    """Tests for compute_eigenvalues function."""

    def test_diagonal_matrix(self):
        """Should return diagonal elements as eigenvalues."""
        J = np.array([[2.0, 0.0], [0.0, 3.0]])

        eigs, vecs = compute_eigenvalues(J)

        assert len(eigs) == 2
        assert pytest.approx(eigs[0], abs=1e-10) in [2.0, 3.0]
        assert pytest.approx(eigs[1], abs=1e-10) in [2.0, 3.0]


class TestCheckStability:
    """Tests for check_stability function."""

    def test_stable_config(self):
        """Should return stable for valid config."""
        config = {"lam": 0.2, "kappa": 0.15, "c1": 0.2, "eta": 0.1}

        result = check_stability(config)

        assert result["stable"] is True
        assert result["max_eigenvalue"] < 1.0

    def test_constraints_satisfied(self):
        """Should check all constraints."""
        config = {"lam": 0.2, "kappa": 0.15, "c1": 0.2, "eta": 0.1}

        result = check_stability(config)

        assert result["constraints_satisfied"]["lambda_positive"] is True
        assert result["constraints_satisfied"]["kappa_positive"] is True


class TestComputeFixedPoint:
    """Tests for compute_fixed_point function."""

    def test_basic_computation(self):
        """Should compute fixed point."""
        config = {"lam": 0.2, "theta_base": 1.0}

        result = compute_fixed_point(config)

        assert "S_star" in result
        assert "theta_star" in result
        assert result["theta_star"] == 1.0


class TestAnalyzeBifurcation:
    """Tests for analyze_bifurcation function."""

    def test_bifurcation_analysis(self):
        """Should analyze bifurcation."""
        config = {"lam": 0.2, "kappa": 0.15, "c1": 0.2, "eta": 0.1}

        result = analyze_bifurcation(
            config,
            param_name="lam",
            param_range=(0.1, 0.5),
            n_points=10,
        )

        assert "parameter_name" in result
        assert "parameter_values" in result
        assert "stability" in result
        assert result["parameter_name"] == "lam"


class TestValidateSystemDynamics:
    """Tests for validate_system_dynamics function."""

    def test_validation_with_data(self):
        """Should validate dynamics."""
        config = {"lam": 0.2, "kappa": 0.15, "c1": 0.2, "eta": 0.1}

        S_history = np.random.normal(1.0, 0.1, 200)
        theta_history = np.random.normal(1.0, 0.1, 200)

        result = validate_system_dynamics(config, S_history, theta_history)

        assert result["valid"] is True
        assert "correlation_S" in result
        assert "correlation_theta" in result

    def test_insufficient_data(self):
        """Should return invalid for insufficient data."""
        config = {"lam": 0.2, "kappa": 0.15, "c1": 0.2, "eta": 0.1}

        S_history = np.random.normal(1.0, 0.1, 50)
        theta_history = np.random.normal(1.0, 0.1, 50)

        result = validate_system_dynamics(config, S_history, theta_history)

        assert result["valid"] is False


class TestStabilityAnalyzer:
    """Tests for StabilityAnalyzer class."""

    def test_initialization(self):
        """Should initialize correctly."""
        config = {"lam": 0.2, "kappa": 0.15, "c1": 0.2, "eta": 0.1}
        analyzer = StabilityAnalyzer(config)

        assert analyzer.config == config
        assert len(analyzer.history["S"]) == 0

    def test_step(self):
        """Should record state."""
        config = {"lam": 0.2, "kappa": 0.15, "c1": 0.2, "eta": 0.1}
        analyzer = StabilityAnalyzer(config)

        analyzer.step(S=1.0, theta=1.0)

        assert len(analyzer.history["S"]) == 1
        assert len(analyzer.history["theta"]) == 1

    def test_analyze(self):
        """Should perform comprehensive analysis."""
        config = {"lam": 0.2, "kappa": 0.15, "c1": 0.2, "eta": 0.1}
        analyzer = StabilityAnalyzer(config)

        # Add some history
        for _ in range(100):
            analyzer.step(
                S=np.random.normal(1.0, 0.1), theta=np.random.normal(1.0, 0.1)
            )

        result = analyzer.analyze()

        assert "stability" in result
        assert "fixed_point" in result
        assert "dynamics_validation" in result
