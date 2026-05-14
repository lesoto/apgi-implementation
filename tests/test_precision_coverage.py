import pytest
import numpy as np
from core.precision import (
    clamp,
    compute_precision,
    update_mean_ema,
    update_variance_ema,
    apply_ach_gain,
    apply_ne_gain,
    apply_dopamine_bias_to_error,
    compute_interoceptive_precision_exponential,
    precision_coupling_ode_core,
    update_precision_euler,
)


def test_clamp():
    assert clamp(5.0, 0.0, 10.0) == 5.0
    assert clamp(-1.0, 0.0, 10.0) == 0.0
    assert clamp(11.0, 0.0, 10.0) == 10.0
    with pytest.raises(ValueError, match="lower must be <= upper"):
        clamp(5.0, 10.0, 0.0)


def test_compute_precision():
    # pi = 1 / (1.0 + 1e-8) ~ 1.0
    assert pytest.approx(compute_precision(1.0)) == 1.0
    # Clamping
    assert compute_precision(1e-10, pi_max=10.0) == 10.0
    # 1/(100 + 1e-8) is ~0.01, so with pi_min=0.1 it clamps to 0.1
    assert compute_precision(100.0, pi_min=0.1) == 0.1


def test_ema_updates():
    # Mean: (1-0.1)*1.0 + 0.1*2.0 = 0.9 + 0.2 = 1.1
    assert pytest.approx(update_mean_ema(1.0, 2.0, 0.1)) == 1.1

    # Var: (1-0.1)*1.0 + 0.1*(2.0-1.0)**2 = 0.9 + 0.1 = 1.0
    assert pytest.approx(update_variance_ema(1.0, 2.0, 1.0, 0.1)) == 1.0

    with pytest.raises(ValueError, match="alpha must be in"):
        update_mean_ema(1.0, 2.0, 1.5)
    with pytest.raises(ValueError, match="alpha must be in"):
        update_variance_ema(1.0, 2.0, 1.0, 0.0)


def test_gains():
    assert apply_ach_gain(10.0, 1.5) == 15.0
    assert apply_ne_gain(10.0, 0.5) == 5.0
    assert apply_dopamine_bias_to_error(0.5, 0.1) == 0.6


def test_interoceptive_precision_exponential():
    # pi = 10 * exp(0.5 * 2) = 10 * e ~ 27.18
    res = compute_interoceptive_precision_exponential(10.0, 0.5, 2.0)
    assert pytest.approx(res) == 10.0 * np.exp(1.0)


def test_precision_coupling_ode_core():
    # lvl 1: pi=1.0, tau=10.0, eps=0.5, alpha=2.0
    # decay = -1/10 = -0.1
    # drive = 2 * 0.5 = 1.0
    # top_down: lvl 2 pi=2.0, C_down=0.1 -> 0.1*(2-1) = 0.1
    # bottom_up: lvl 0 eps=0.2, C_up=0.5, psi=abs -> 0.5*0.2 = 0.1
    # total = -0.1 + 1.0 + 0.1 + 0.1 = 1.1
    dpi = precision_coupling_ode_core(
        pi_ell=1.0,
        tau_pi=10.0,
        epsilon_ell=0.5,
        alpha_gain=2.0,
        pi_ell_plus_1=2.0,
        epsilon_ell_minus_1=0.2,
        C_down=0.1,
        C_up=0.5,
        psi=abs,
    )
    assert pytest.approx(dpi) == 1.1

    # Boundary (None)
    dpi_boundary = precision_coupling_ode_core(
        pi_ell=1.0,
        tau_pi=10.0,
        epsilon_ell=0.5,
        alpha_gain=2.0,
        pi_ell_plus_1=None,
        epsilon_ell_minus_1=None,
        C_down=0.1,
        C_up=0.5,
    )
    assert pytest.approx(dpi_boundary) == -0.1 + 1.0


def test_update_precision_euler():
    # pi=1.0, dpi=1.1, dt=0.1 -> 1.0 + 0.11 = 1.11
    assert pytest.approx(update_precision_euler(1.0, 1.1, 0.1)) == 1.11
