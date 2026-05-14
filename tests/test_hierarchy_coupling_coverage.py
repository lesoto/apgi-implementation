import pytest
import numpy as np
from hierarchy.coupling import (
    estimate_hierarchy_levels,
    precision_coupling_ode,
    phase_locked_threshold,
    nonlinear_phase_amplitude_coupling,
    bidirectional_phase_coupling,
    bottom_up_threshold_cascade,
    bidirectional_threshold_cascade,
    update_phase_kuramoto_full,
    HierarchicalPrecisionNetwork,
)


def test_estimate_hierarchy_levels():
    assert estimate_hierarchy_levels(0.01, 10.0, 1.6) == 15
    with pytest.raises(ValueError, match="must be > 0"):
        estimate_hierarchy_levels(0, 10.0)
    with pytest.raises(ValueError, match="k > 1"):
        estimate_hierarchy_levels(0.01, 10.0, 1.0)


def test_precision_coupling_ode():
    # Base case
    val = precision_coupling_ode(
        pi_ell=1.0,
        tau_pi=100.0,
        epsilon_ell=0.5,
        alpha_gain=0.1,
        pi_ell_plus_1=1.2,
        epsilon_ell_minus_1=0.4,
        C_down=0.1,
        C_up=0.05,
    )
    assert isinstance(val, float)

    # Top level (no pi_ell_plus_1)
    precision_coupling_ode(1.0, 100.0, 0.5, 0.1, None, 0.4, 0.1, 0.05)

    # Bottom level (no epsilon_ell_minus_1)
    precision_coupling_ode(1.0, 100.0, 0.5, 0.1, 1.2, None, 0.1, 0.05)

    # With psi function
    val_psi = precision_coupling_ode(1.0, 100.0, 0.5, 0.1, 1.2, 0.4, 0.1, 0.05, psi=lambda x: x**2)
    assert val_psi != val


def test_phase_locked_threshold():
    th = phase_locked_threshold(1.0, 1.0, 0.0, 0.1)  # cos(0)=1
    assert th == pytest.approx(1.1)

    # With phase_sensitivity
    th_s = phase_locked_threshold(1.0, 1.0, 0.0, 0.1, phase_sensitivity=0.5)
    assert th_s == pytest.approx(1.05)


def test_nonlinear_phase_amplitude_coupling():
    th0 = 1.0
    pi = 1.0
    phi = 0.0
    k = 0.1

    # Sigmoid
    th_sig = nonlinear_phase_amplitude_coupling(th0, pi, phi, k, nonlinearity="sigmoid")
    pi_sig = 1.0 / (1.0 + np.exp(-1.0))
    assert th_sig == pytest.approx(th0 * (1.0 + k * pi_sig))

    # Power
    th_pow = nonlinear_phase_amplitude_coupling(th0, pi, phi, k, nonlinearity="power")
    assert th_pow == pytest.approx(1.1)

    # Exponential
    th_exp = nonlinear_phase_amplitude_coupling(th0, pi, phi, k, nonlinearity="exponential")
    pi_exp = np.exp(1.0) - 1.0
    assert th_exp == pytest.approx(th0 * (1.0 + k * pi_exp))

    # Default
    th_def = nonlinear_phase_amplitude_coupling(th0, pi, phi, k, nonlinearity="none")
    assert th_def == pytest.approx(1.1)


def test_bidirectional_phase_coupling():
    phi = 1.0
    phi_up = 1.2
    phi_down = 0.8
    omega = 0.1
    dt = 1.0

    # All couplings
    p_new = bidirectional_phase_coupling(phi, phi_up, phi_down, omega, dt)
    assert 0 <= p_new < 2 * np.pi

    # With noise
    rng = np.random.default_rng(42)
    bidirectional_phase_coupling(phi, None, None, omega, dt, noise_std=0.1, rng=rng)


def test_bottom_up_threshold_cascade():
    th = bottom_up_threshold_cascade(1.0, 1.5, 1.0, 0.1)  # superthreshold
    assert th == pytest.approx(0.9)

    th_below = bottom_up_threshold_cascade(1.0, 0.5, 1.0, 0.1)  # below
    assert th_below == pytest.approx(1.0)


def test_bidirectional_threshold_cascade():
    th = 1.0
    s_m = 1.5
    t_m = 1.0  # lower super
    s_p = 1.5
    t_p = 1.0  # upper super

    # Both active
    th_bi = bidirectional_threshold_cascade(th, s_m, t_m, s_p, t_p, hysteresis=0)
    assert th_bi == pytest.approx(1.0 * (1.0 - 0.1) * (1.0 + 0.05))

    # None active
    th_none = bidirectional_threshold_cascade(th, None, None, None, None)
    assert th_none == 1.0


def test_update_phase_kuramoto_full():
    phis = np.array([0.1, 0.2])
    omegas = np.array([0.1, 0.1])
    K = np.array([[0, 0.1], [0.1, 0]])
    p_new = update_phase_kuramoto_full(phis, omegas, K, dt=1.0, noise_std=0.1)
    assert len(p_new) == 2


def test_hierarchical_precision_network():
    net = HierarchicalPrecisionNetwork(n_levels=3)
    eps = np.array([0.1, 0.2, 0.3])
    pi, phi = net.step(eps, dt=1.0)
    assert len(pi) == 3
    assert len(phi) == 3

    # Thresholds
    th0 = np.ones(3)
    thetas = net.compute_thresholds(th0, S_levels=np.array([0.5, 1.5, 0.5]), kappa_up=0.1)
    assert len(thetas) == 3

    # Custom taus
    net2 = HierarchicalPrecisionNetwork(n_levels=2, taus=[10, 100])
    assert net2.taus[0] == 10


def test_update_phase_dynamics_noise():
    from hierarchy.coupling import update_phase_dynamics

    rng = np.random.default_rng(42)
    p = update_phase_dynamics(0.0, 0.1, 1.0, noise_std=0.5, rng=rng)
    assert p != 0.1  # Should be different due to noise
