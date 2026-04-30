"""Extended tests for validation/observable_mapping.py to achieve 100% coverage."""

import numpy as np

from validation.observable_mapping import ParameterIdentifiabilityAnalyzer


class TestParameterIdentifiabilityAnalyzerExtended:
    """Extended tests for ParameterIdentifiabilityAnalyzer class."""

    def test_compute_fisher_matrix_identifiability_error(self):
        """Should handle singular Fisher matrix."""
        # Create degenerate history that makes Fisher matrix singular
        # All identical gradients
        S_history = np.ones(100)
        theta_history = np.ones(100)
        B_history = np.zeros(100)
        params = {
            "lam": 0.2,
            "eta": 0.1,
            "ignite_tau": 0.5,
        }
        result = ParameterIdentifiabilityAnalyzer.compute_fisher_information(
            S_history, theta_history, B_history, params
        )
        # Should return very large condition number and non-identifiable
        assert result["condition_number"] > 1e6
        assert not result["identifiable"]  # Use not to handle numpy bool

    def test_check_identifiability_constraints(self):
        """Should check identifiability constraints."""
        config = {
            "lam": 0.2,
            "tau_s": 5.0,
            "eta": 0.1,
            "delta": 0.5,
            "ignite_tau": 0.5,
        }
        result = ParameterIdentifiabilityAnalyzer.check_identifiability_constraints(config)
        # Should return dict with constraint results
        assert isinstance(result, dict)

    def test_compute_fisher_matrix_linalg_error(self, suppress_lapack):
        """Should handle LinAlgError in Fisher matrix computation."""
        # Create data that will cause LinAlgError
        S_history = np.array([1.0, 1.0, 1.0])  # Constant signal
        theta_history = np.array([1.0, 1.0, 1.0])
        B_history = np.array([0, 0, 0])
        params = {"lam": 0.2, "eta": 0.1, "ignite_tau": 0.5}
        result = ParameterIdentifiabilityAnalyzer.compute_fisher_information(
            S_history, theta_history, B_history, params
        )
        # Should handle LinAlgError and set condition_number to a very large value
        assert result["condition_number"] > 1e6
