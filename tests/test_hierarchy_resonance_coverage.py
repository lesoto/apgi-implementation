import pytest
import numpy as np
from hierarchy.resonance import NestedResonanceSystem, build_resonance_system, LevelState


def test_level_state():
    # Just to exercise the dataclass
    ls = LevelState(S=1.0, theta=1.0, phi=0.0, pi=1.0)
    assert ls.S == 1.0


def test_resonance_system_init():
    n_levels = 3
    theta_0 = [1.0, 1.0, 1.0]
    omega = [0.1, 0.05, 0.025]
    lambda_rates = [0.5, 0.2, 0.1]

    sys = NestedResonanceSystem(n_levels, theta_0, omega, lambda_rates)
    assert sys.n_levels == 3
    assert sys.S.shape == (3,)

    with pytest.raises(ValueError, match="n_levels"):
        NestedResonanceSystem(0, theta_0, omega, lambda_rates)


def test_resonance_system_step():
    n_levels = 3
    sys = build_resonance_system(n_levels, np.array([10, 20, 40]), 1.0, 1.0, phi_noise_std=0.1)

    s_inst = np.array([0.5, 0.5, 0.5])
    pi = np.array([1.0, 1.0, 1.0])

    # Take a few steps
    for _ in range(5):
        sys.step(s_inst, pi, dt=1.0)

    assert sys.primary_signal > 0
    assert 0 <= sys.phi[0] < 2 * np.pi
    assert sys.primary_threshold > 0

    # Ignition windows
    sys.S[0] = 5.0
    sys.theta[0] = 1.0
    assert sys.ignition_windows[0]

    # Modulation depth
    depth = sys.modulation_depth
    assert len(depth) == 3
    assert depth[2] == 0.0  # Top level


def test_apply_level_ignition():
    sys = build_resonance_system(3, np.array([10, 20, 40]), 1.0, 1.0, kappa_up=0.1)
    sys.S[0] = 1.0
    th0 = sys.theta[1]

    sys.apply_level_ignition(0, rho_S=0.1, delta_refractory=0.5)
    assert sys.S[0] == pytest.approx(0.1)
    assert sys.theta[1] == pytest.approx(th0 * 0.9)

    with pytest.raises(ValueError, match="out of range"):
        sys.apply_level_ignition(5)


def test_resonance_system_factory():
    sys = build_resonance_system(2, np.array([10, 100]), 1.0, 1.0)
    assert sys.n_levels == 2
    assert sys.lambda_rates[0] == pytest.approx(0.1)
    assert sys.lambda_rates[1] == pytest.approx(0.01)
