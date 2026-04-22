"""Extended tests for core/preprocessing.py to achieve 100% coverage."""

import pytest
from core.preprocessing import (
    update_prediction,
)


class TestUpdatePrediction:
    """Extended tests for update_prediction function."""

    def test_prediction_update_formula(self):
        """Should update prediction correctly with given formula."""
        x_hat = 1.0
        epsilon = 0.5
        pi = 2.0
        kappa = 0.1
        result = update_prediction(x_hat, epsilon, pi, kappa)
        # x̂(t+1) = x̂(t) + κ · Π(t) · ε(t)
        # = 1.0 + 0.1 * 2.0 * 0.5 = 1.0 + 0.1 = 1.1
        expected = 1.0 + 0.1 * 2.0 * 0.5
        assert result == pytest.approx(expected)

    def test_prediction_update_with_zero_kappa(self):
        """Should not change prediction when kappa is 0."""
        x_hat = 1.0
        epsilon = 0.5
        pi = 2.0
        kappa = 0.0
        result = update_prediction(x_hat, epsilon, pi, kappa)
        assert result == x_hat

    def test_prediction_update_with_zero_pi(self):
        """Should not change prediction when pi is 0."""
        x_hat = 1.0
        epsilon = 0.5
        pi = 0.0
        kappa = 0.1
        result = update_prediction(x_hat, epsilon, pi, kappa)
        assert result == x_hat

    def test_prediction_update_with_zero_epsilon(self):
        """Should not change prediction when epsilon is 0."""
        x_hat = 1.0
        epsilon = 0.0
        pi = 2.0
        kappa = 0.1
        result = update_prediction(x_hat, epsilon, pi, kappa)
        assert result == x_hat
