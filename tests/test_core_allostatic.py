"""Comprehensive unit tests for core/allostatic.py module.

Tests cover:
- allostatic_threshold_ode function
- update_threshold_euler function
- AllostaticThresholdController class
"""

from __future__ import annotations

import numpy as np
import pytest

from core.allostatic import (
    AllostaticThresholdController,
    allostatic_threshold_ode,
    update_threshold_euler,
)


class TestAllostaticThresholdODE:
    """Tests for allostatic_threshold_ode function."""

    def test_basic_ode(self):
        """Should compute ODE update correctly."""
        np.random.seed(42)
        result = allostatic_threshold_ode(
            theta=1.0,
            theta_0=1.0,
            gamma=0.01,
            B_prev=0,
            delta=0.5,
            C=1.5,
            V=1.0,
            eta=0.1,
            dt=1.0,
            noise_std=0.0,
        )
        # Mean reversion: -0.01*(1.0-1.0) = 0
        # Refractory: 0.5*0 = 0
        # Allostatic: 0.1*(1.5-1.0) = 0.05
        # Total drift: 0.05, theta_new = 1.0 + 0.05*1.0 + 0 = 1.05
        assert pytest.approx(result, rel=1e-6) == 1.05

    def test_mean_reversion(self):
        """Should revert to baseline."""
        np.random.seed(42)
        result = allostatic_threshold_ode(
            theta=2.0,
            theta_0=1.0,
            gamma=0.1,
            B_prev=0,
            delta=0.5,
            C=1.0,
            V=1.0,
            eta=0.0,
            dt=1.0,
            noise_std=0.0,
        )
        # Should move toward baseline
        assert result < 2.0
        assert result > 1.0

    def test_refractory_boost(self):
        """Should add refractory boost after ignition."""
        np.random.seed(42)
        result_with = allostatic_threshold_ode(
            theta=1.0,
            theta_0=1.0,
            gamma=0.01,
            B_prev=1,
            delta=0.5,
            C=1.0,
            V=1.0,
            eta=0.0,
            dt=1.0,
            noise_std=0.0,
        )

        np.random.seed(42)
        result_without = allostatic_threshold_ode(
            theta=1.0,
            theta_0=1.0,
            gamma=0.01,
            B_prev=0,
            delta=0.5,
            C=1.0,
            V=1.0,
            eta=0.0,
            dt=1.0,
            noise_std=0.0,
        )

        assert result_with > result_without

    def test_cost_value_mismatch(self):
        """Should respond to cost-value mismatch."""
        np.random.seed(42)
        result_high_cost = allostatic_threshold_ode(
            theta=1.0,
            theta_0=1.0,
            gamma=0.01,
            B_prev=0,
            delta=0.5,
            C=2.0,
            V=1.0,
            eta=0.1,
            dt=1.0,
            noise_std=0.0,
        )

        np.random.seed(42)
        result_high_value = allostatic_threshold_ode(
            theta=1.0,
            theta_0=1.0,
            gamma=0.01,
            B_prev=0,
            delta=0.5,
            C=1.0,
            V=2.0,
            eta=0.1,
            dt=1.0,
            noise_std=0.0,
        )

        assert result_high_cost > result_high_value

    def test_stochastic_noise(self):
        """Should add stochastic noise."""
        np.random.seed(42)
        results = [
            allostatic_threshold_ode(
                theta=1.0,
                theta_0=1.0,
                gamma=0.01,
                B_prev=0,
                delta=0.5,
                C=1.0,
                V=1.0,
                eta=0.0,
                dt=1.0,
                noise_std=0.1,
            )
            for _ in range(10)
        ]

        # Results should vary due to noise
        assert len(set([round(r, 6) for r in results])) > 1


class TestUpdateThresholdEuler:
    """Tests for update_threshold_euler function."""

    def test_basic_update(self):
        """Should compute Euler update correctly."""
        result = update_threshold_euler(
            theta=1.0,
            theta_dot=0.1,
            dt=1.0,
        )
        assert result == 1.1

    def test_clamping_min(self):
        """Should clamp to minimum."""
        result = update_threshold_euler(
            theta=0.05,
            theta_dot=-0.1,
            dt=1.0,
            theta_min=0.1,
        )
        assert result == 0.1

    def test_clamping_max(self):
        """Should clamp to maximum."""
        result = update_threshold_euler(
            theta=999.0,
            theta_dot=100.0,
            dt=1.0,
            theta_max=1000.0,
        )
        assert result == 1000.0

    def test_negative_derivative(self):
        """Should decrease threshold with negative derivative."""
        result = update_threshold_euler(
            theta=1.0,
            theta_dot=-0.1,
            dt=1.0,
        )
        assert result == 0.9


class TestAllostaticThresholdController:
    """Tests for AllostaticThresholdController class."""

    def test_initialization(self):
        """Should initialize with correct parameters."""
        controller = AllostaticThresholdController(
            theta_0=1.5,
            gamma=0.02,
            delta=0.6,
            dt=0.5,
        )

        assert controller.theta == 1.5
        assert controller.theta_0 == 1.5
        assert controller.gamma == 0.02
        assert controller.delta == 0.6
        assert controller.dt == 0.5
        assert controller.B_prev == 0

    def test_step_basic(self):
        """Should update threshold on step."""
        np.random.seed(42)
        controller = AllostaticThresholdController(
            theta_0=1.0,
            gamma=0.01,
            delta=0.5,
            dt=1.0,
        )

        result = controller.step(
            C=1.5,
            V=1.0,
            eta=0.1,
            B=0,
            noise_std=0.0,
        )

        # Should have updated theta
        assert result != 1.0
        assert controller.theta == result

    def test_step_stores_B(self):
        """Should store ignition state for next step."""
        controller = AllostaticThresholdController(theta_0=1.0, gamma=0.01, delta=0.5, dt=1.0)

        controller.step(C=1.0, V=1.0, eta=0.0, B=1, noise_std=0.0)
        assert controller.B_prev == 1

        controller.step(C=1.0, V=1.0, eta=0.0, B=0, noise_std=0.0)
        assert controller.B_prev == 0

    def test_reset(self):
        """Should reset to baseline."""
        np.random.seed(42)
        controller = AllostaticThresholdController(theta_0=1.0, gamma=0.01, delta=0.5, dt=1.0)

        # Step to change theta
        controller.step(C=2.0, V=1.0, eta=0.1, B=0, noise_std=0.0)
        assert controller.theta != 1.0

        # Reset
        controller.reset()
        assert controller.theta == 1.0
        assert controller.B_prev == 0

    def test_reset_with_custom_value(self):
        """Should reset to custom value."""
        controller = AllostaticThresholdController(theta_0=1.0, gamma=0.01, delta=0.5, dt=1.0)
        controller.reset(theta=2.5)
        assert controller.theta == 2.5

    def test_multiple_steps(self):
        """Should maintain state across multiple steps."""
        np.random.seed(42)
        controller = AllostaticThresholdController(theta_0=1.0, gamma=0.01, delta=0.5, dt=1.0)

        thetas = []
        for i in range(10):
            theta = controller.step(
                C=1.5,
                V=1.0,
                eta=0.1,
                B=1 if i == 5 else 0,
                noise_std=0.0,
            )
            thetas.append(theta)

        # Should have history of different values
        assert len(set([round(t, 6) for t in thetas])) > 1
