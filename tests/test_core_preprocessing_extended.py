"""Extended tests for core/preprocessing.py to achieve 100% coverage."""

import pytest

from core.preprocessing import EMAStats, update_prediction


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


class TestEMAStats:
    """Tests for EMAStats class to cover missing lines."""

    def test_init_valid_alpha(self):
        """Should initialize with valid alpha value."""
        ema = EMAStats(alpha=0.5, initial_mean=1.0, initial_var=2.0)
        assert ema.alpha == 0.5
        assert ema._mean == 1.0
        assert ema._var == 2.0

    def test_init_alpha_zero_raises(self):
        """Should raise ValueError when alpha is 0."""
        with pytest.raises(ValueError, match="alpha must be in"):
            EMAStats(alpha=0.0)

    def test_init_alpha_negative_raises(self):
        """Should raise ValueError when alpha is negative."""
        with pytest.raises(ValueError, match="alpha must be in"):
            EMAStats(alpha=-0.1)

    def test_init_alpha_one_valid(self):
        """Should accept alpha=1.0 (boundary value)."""
        ema = EMAStats(alpha=1.0)
        assert ema.alpha == 1.0

    def test_update_mean_calculation(self):
        """Should update mean correctly using EMA formula."""
        ema = EMAStats(alpha=0.5, initial_mean=1.0)
        ema.update(2.0)
        # μ(t+1) = (1-α)μ(t) + α·z(t) = 0.5*1.0 + 0.5*2.0 = 1.5
        expected = 0.5 * 1.0 + 0.5 * 2.0
        assert ema._mean == pytest.approx(expected)

    def test_update_variance_calculation(self):
        """Should update variance correctly using centered deviation."""
        ema = EMAStats(alpha=0.5, initial_mean=1.0, initial_var=1.0)
        ema.update(2.0)
        # After mean update: mean = 1.5
        # Variance uses new mean: (value - new_mean)^2 = (2 - 1.5)^2 = 0.25
        # σ²(t+1) = (1-α)σ²(t) + α·(z-μ)² = 0.5*1.0 + 0.5*0.25 = 0.625
        assert ema._var == pytest.approx(0.625)

    def test_mean_returns_float(self):
        """Should return mean as float."""
        ema = EMAStats(alpha=0.5, initial_mean=3.14)
        result = ema.mean()
        assert isinstance(result, float)
        assert result == pytest.approx(3.14)

    def test_variance_returns_float(self):
        """Should return variance as float."""
        ema = EMAStats(alpha=0.5, initial_var=2.71)
        result = ema.variance()
        assert isinstance(result, float)
        assert result == pytest.approx(2.71)

    def test_std_calculation(self):
        """Should compute std correctly from variance."""
        ema = EMAStats(alpha=0.5, initial_var=4.0)
        result = ema.std()
        assert result == pytest.approx(2.0)

    def test_std_with_zero_variance(self):
        """Should handle zero variance."""
        ema = EMAStats(alpha=0.5, initial_var=0.0)
        result = ema.std()
        assert result == 0.0

    def test_std_with_negative_variance_clamped(self):
        """Should clamp negative variance to zero via max()."""
        ema = EMAStats(alpha=0.5, initial_var=-1.0)
        result = ema.std()
        # max(-1.0, 0.0) = 0.0, sqrt(0.0) = 0.0
        assert result == 0.0

    def test_ema_full_sequence(self):
        """Should maintain correct stats through multiple updates."""
        ema = EMAStats(alpha=0.3, initial_mean=0.0, initial_var=1.0)
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        for v in values:
            ema.update(v)
        # After multiple updates, verify we have reasonable values
        assert ema.mean() > 0
        assert ema.variance() >= 0
        assert ema.std() >= 0
