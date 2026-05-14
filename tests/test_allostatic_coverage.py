import pytest
from core.allostatic import (
    allostatic_threshold_ode,
    update_threshold_euler,
    AllostaticThresholdController,
)


def test_allostatic_threshold_ode():
    # Deterministic: theta=1.0, theta_0=1.0, gamma=0.1, B_prev=1, delta=0.5, C=1.0, V=0.0, eta=0.1, dt=1.0
    # reversion = -0.1 * (1-1) = 0
    # refractory = 0.5 * 1 = 0.5
    # allostatic = 0.1 * (1-0) = 0.1
    # drift = 0.6
    # theta_new = 1.0 + 0.6*1.0 = 1.6
    res = allostatic_threshold_ode(
        1.0, 1.0, 0.1, 1, 0.5, C=1.0, V=0.0, eta=0.1, dt=1.0, noise_std=0.0
    )
    assert pytest.approx(res) == 1.6


def test_update_threshold_euler():
    # 1.0 + 1.0 * 0.5 = 1.5
    assert update_threshold_euler(1.0, 0.5, dt=1.0) == 1.5
    # Clamping
    assert update_threshold_euler(1.0, -2.0, dt=1.0, theta_min=0.5) == 0.5
    assert update_threshold_euler(100.0, 100.0, dt=1.0, theta_max=150.0) == 150.0


def test_allostatic_threshold_controller():
    ctrl = AllostaticThresholdController(theta_0=1.0, gamma=0.1, delta=0.5, dt=1.0)
    assert ctrl.theta == 1.0
    assert ctrl.B_prev == 0

    # Step 1: B=1. C=1.0, V=0.0, eta=0.1
    # B_prev is 0 initially.
    # reversion=0, refractory=0, allostatic=0.1 -> drift=0.1 -> theta=1.1
    # B_prev becomes 1.
    new_theta = ctrl.step(C=1.0, V=0.0, eta=0.1, B=1, noise_std=0.0)
    assert pytest.approx(new_theta) == 1.1
    assert ctrl.B_prev == 1

    # Step 2:
    # reversion = -0.1*(1.1-1.0) = -0.01
    # refractory = 0.5 * 1 = 0.5
    # allostatic = 0.1 * (1.0-0.0) = 0.1
    # drift = 0.59
    # theta_new = 1.1 + 0.59 = 1.69
    new_theta_2 = ctrl.step(C=1.0, V=0.0, eta=0.1, B=0, noise_std=0.0)
    assert pytest.approx(new_theta_2) == 1.69

    # Reset
    ctrl.reset(theta=2.0)
    assert ctrl.theta == 2.0
    assert ctrl.B_prev == 0

    ctrl.reset()
    assert ctrl.theta == 1.0
