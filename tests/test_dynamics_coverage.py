import pytest
import numpy as np
from core.dynamics import (
    signal_drift,
    update_signal_ode,
    compute_precision_coupled_noise_std,
    update_prediction,
    update_threshold_ode,
)


def test_signal_drift():
    # S=1.0, z_e=0.5, z_i=0.5, pi_e=2.0, pi_i=2.0, beta=0.0, tau_s=10.0
    # dS/dt = -1/10 + 2*0.5 + 2*(0.5+0) = -0.1 + 1 + 1 = 1.9
    drift = signal_drift(1.0, 0.5, 0.5, 2.0, 2.0, 0.0, 10.0)
    assert drift == 1.9

    with pytest.raises(ValueError, match="tau_s must be > 0"):
        signal_drift(1.0, 0.5, 0.5, 2.0, 2.0, 0.0, 0.0)


def test_update_signal_ode():
    # No noise case
    # S=1.0, drift=1.9, dt=0.1 -> S_new = 1.0 + 1.9*0.1 = 1.19
    s_new = update_signal_ode(1.0, 0.5, 0.5, 2.0, 2.0, 0.0, 10.0, dt=0.1, noise_std=0.0)
    assert pytest.approx(s_new) == 1.19

    with pytest.raises(ValueError, match="tau_s must be > 0"):
        update_signal_ode(1.0, 0.5, 0.5, 2.0, 2.0, 0.0, 0.0)


def test_compute_precision_coupled_noise_std():
    assert compute_precision_coupled_noise_std(2.0, 2.0) == 1.0 / np.sqrt(4.0)
    assert compute_precision_coupled_noise_std(0.0, 0.0) == 1.0


def test_update_prediction():
    # x_hat=0.0, eps=1.0, pi=10.0, kappa=0.1
    # x_new = 0.0 + 0.1 * 10.0 * 1.0 = 1.0
    assert update_prediction(0.0, 1.0, 10.0, 0.1) == 1.0


def test_update_threshold_ode():
    # No noise case
    # theta=1.0, base=1.0, C=1.0, V=0.0, tau=10.0, eta=0.1, delta=0.5, B=1, dt=1.0
    # decay = -(1-1)/10 = 0
    # allostatic = 0.1 * (1-0) = 0.1
    # refractory = 0.5 * 1 = 0.5
    # drift = 0.6
    # theta_new = 1.0 + 0.6 * 1.0 = 1.6
    t_new = update_threshold_ode(
        1.0, 1.0, 1.0, 0.0, 10.0, 0.1, delta=0.5, B=1, dt=1.0, noise_std=0.0
    )
    assert pytest.approx(t_new) == 1.6

    with pytest.raises(ValueError, match="tau_theta must be > 0"):
        update_threshold_ode(1.0, 1.0, 1.0, 0.0, 0.0, 0.1)
