"""Comprehensive unit tests for core/threshold.py module.

Tests cover:
- compute_metabolic_cost function
- compute_metabolic_cost_realistic function
- compute_information_value function
- apply_ne_threshold_modulation function
- threshold_decay function
- update_threshold_discrete function
- apply_refractory_boost function
"""

from __future__ import annotations

import numpy as np
import pytest

from core.threshold import (
    compute_metabolic_cost,
    compute_metabolic_cost_realistic,
    compute_information_value,
    apply_ne_threshold_modulation,
    threshold_decay,
    update_threshold_discrete,
    apply_refractory_boost,
)


class TestComputeMetabolicCost:
    """Tests for compute_metabolic_cost function."""

    def test_linear_cost(self):
        """Should compute linear cost C(t) = c0 + c1*S(t)."""
        result = compute_metabolic_cost(S=2.0, c0=0.5, c1=0.5)
        assert result == 1.5

    def test_zero_c0(self):
        """Should handle zero baseline cost."""
        result = compute_metabolic_cost(S=2.0, c0=0.0, c1=0.5)
        assert result == 1.0

    def test_zero_signal(self):
        """Should return c0 when signal is zero."""
        result = compute_metabolic_cost(S=0.0, c0=0.5, c1=0.5)
        assert result == 0.5


class TestComputeMetabolicCostRealistic:
    """Tests for compute_metabolic_cost_realistic function."""

    def test_realistic_cost(self):
        """Should compute realistic cost C(t) = c1*S(t) + c2*B(t-1)."""
        result = compute_metabolic_cost_realistic(S=2.0, B_prev=1, c1=0.5, c2=1.0)
        assert result == 2.0  # 0.5*2.0 + 1.0*1 = 2.0

    def test_no_previous_ignition(self):
        """Should exclude ignition cost when B_prev=0."""
        result = compute_metabolic_cost_realistic(S=2.0, B_prev=0, c1=0.5, c2=1.0)
        assert result == 1.0  # 0.5*2.0 + 0 = 1.0

    def test_previous_ignition(self):
        """Should add ignition cost when B_prev=1."""
        result = compute_metabolic_cost_realistic(S=2.0, B_prev=1, c1=0.5, c2=1.0)
        assert result == 2.0  # 0.5*2.0 + 1.0 = 2.0


class TestComputeInformationValue:
    """Tests for compute_information_value function."""

    def test_basic_value(self):
        """Should compute value V(t) = v1|z_e| + v2|z_i_eff|."""
        result = compute_information_value(
            z_e=1.0,
            z_i_eff=0.5,
            v1=0.5,
            v2=0.5,
        )
        assert result == 0.75  # 0.5*1.0 + 0.5*0.5 = 0.75

    def test_negative_errors(self):
        """Should use absolute values."""
        result = compute_information_value(
            z_e=-1.0,
            z_i_eff=-0.5,
            v1=0.5,
            v2=0.5,
        )
        assert result == 0.75

    def test_zero_weights(self):
        """Should return zero when weights are zero."""
        result = compute_information_value(1.0, 0.5, v1=0.0, v2=0.0)
        assert result == 0.0


class TestApplyNeThresholdModulation:
    """Tests for apply_ne_threshold_modulation function."""

    def test_ne_modulation(self):
        """Should apply NE modulation θ <- θ * (1 + γ_NE * g_NE)."""
        result = apply_ne_threshold_modulation(theta=1.0, g_ne=0.5, gamma_ne=0.1)
        # θ = 1.0 * (1 + 0.1 * 0.5) = 1.0 * 1.05 = 1.05
        assert result == 1.05

    def test_zero_gain(self):
        """Should return original theta when g_NE=0."""
        result = apply_ne_threshold_modulation(theta=1.0, g_ne=0.0, gamma_ne=0.1)
        assert result == 1.0

    def test_increase_threshold(self):
        """Should increase threshold for positive gain."""
        result = apply_ne_threshold_modulation(theta=1.0, g_ne=1.0, gamma_ne=0.1)
        assert result > 1.0

    def test_decrease_threshold(self):
        """Should decrease threshold for negative gain."""
        result = apply_ne_threshold_modulation(theta=1.0, g_ne=-1.0, gamma_ne=0.1)
        assert result < 1.0


class TestThresholdDecay:
    """Tests for threshold_decay function."""

    def test_exponential_decay(self):
        """Should compute exponential decay correctly."""
        result = threshold_decay(theta=2.0, theta_base=1.0, kappa=0.5)
        expected = 1.0 + (2.0 - 1.0) * np.exp(-0.5)
        assert pytest.approx(result, rel=1e-7) == expected

    def test_already_at_base(self):
        """Should return theta_base when theta equals base."""
        result = threshold_decay(theta=1.0, theta_base=1.0, kappa=0.5)
        assert result == 1.0

    def test_high_kappa_fast_decay(self):
        """Should decay faster with higher kappa."""
        result_fast = threshold_decay(theta=2.0, theta_base=1.0, kappa=2.0)
        result_slow = threshold_decay(theta=2.0, theta_base=1.0, kappa=0.1)
        assert result_fast < result_slow

    def test_zero_kappa(self):
        """Should return original theta when kappa=0."""
        result = threshold_decay(theta=2.0, theta_base=1.0, kappa=0.0)
        assert result == 2.0

    def test_invalid_kappa(self):
        """Should raise ValueError for negative kappa."""
        with pytest.raises(ValueError, match="kappa must be >= 0"):
            threshold_decay(theta=2.0, theta_base=1.0, kappa=-0.1)


class TestUpdateThresholdDiscrete:
    """Tests for update_threshold_discrete function."""

    def test_basic_update(self):
        """Should compute threshold update correctly."""
        result = update_threshold_discrete(
            theta=1.0,
            metabolic_cost=1.5,
            information_value=1.0,
            eta=0.1,
            delta=0.5,
            B_prev=0,
        )
        # θ = 1.0 + 0.1*(1.5-1.0) + 0.5*0 = 1.0 + 0.05 = 1.05
        assert result == 1.05

    def test_with_ignition(self):
        """Should add refractory boost after ignition."""
        result = update_threshold_discrete(
            theta=1.0,
            metabolic_cost=1.5,
            information_value=1.0,
            eta=0.1,
            delta=0.5,
            B_prev=1,
        )
        # θ = 1.0 + 0.1*(1.5-1.0) + 0.5*1 = 1.0 + 0.05 + 0.5 = 1.55
        assert result == 1.55

    def test_cost_greater_than_value(self):
        """Should increase threshold when cost > value."""
        result = update_threshold_discrete(
            theta=1.0,
            metabolic_cost=2.0,
            information_value=1.0,
            eta=0.1,
            delta=0.5,
            B_prev=0,
        )
        assert result > 1.0

    def test_value_greater_than_cost(self):
        """Should decrease threshold when value > cost."""
        result = update_threshold_discrete(
            theta=1.0,
            metabolic_cost=1.0,
            information_value=2.0,
            eta=0.1,
            delta=0.5,
            B_prev=0,
        )
        assert result < 1.0


class TestApplyRefractoryBoost:
    """Tests for apply_refractory_boost function."""

    def test_refractory_boost(self):
        """Should add refractory boost after ignition."""
        result = apply_refractory_boost(theta_next=1.0, B=1, delta=0.5)
        assert result == 1.5

    def test_no_boost_without_ignition(self):
        """Should not add boost when no ignition."""
        result = apply_refractory_boost(theta_next=1.0, B=0, delta=0.5)
        assert result == 1.0

    def test_boost_with_integer_B(self):
        """Should handle B as integer."""
        result = apply_refractory_boost(theta_next=1.0, B=1, delta=0.5)
        assert result == 1.5

    def test_zero_delta(self):
        """Should not change theta when delta=0."""
        result = apply_refractory_boost(theta_next=1.0, B=1, delta=0.0)
        assert result == 1.0
