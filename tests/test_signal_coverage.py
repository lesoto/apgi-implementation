import pytest
import numpy as np
from core.signal import (
    instantaneous_signal,
    instantaneous_signal_with_dopamine,
    integrate_signal_leaky,
    stabilize_signal_log,
    instantaneous_signal_phi,
    compute_apgi_signal,
)


def test_instantaneous_signal():
    # 2*abs(0.5) + 3*abs(-0.2) = 1.0 + 0.6 = 1.6
    assert pytest.approx(instantaneous_signal(0.5, -0.2, 2.0, 3.0)) == 1.6


def test_instantaneous_signal_with_dopamine():
    # 2*0.5 + 3*0.2 + 0.1 = 1.0 + 0.6 + 0.1 = 1.7
    assert pytest.approx(instantaneous_signal_with_dopamine(0.5, 0.2, 2.0, 3.0, 0.1)) == 1.7


def test_integrate_signal_leaky():
    # (1-0.2)*1.0 + 0.2*2.0 = 0.8 + 0.4 = 1.2
    assert pytest.approx(integrate_signal_leaky(1.0, 2.0, 0.2)) == 1.2
    with pytest.raises(ValueError, match="lam must be in"):
        integrate_signal_leaky(1.0, 2.0, 0.0)


def test_stabilize_signal_log():
    assert stabilize_signal_log(1.0, enabled=False) == 1.0
    assert stabilize_signal_log(np.e - 1.0, enabled=True) == pytest.approx(1.0)
    assert stabilize_signal_log(-1.0, enabled=True) == 0.0  # max(0, -1) = 0, log1p(0) = 0


def test_instantaneous_signal_phi():
    # 2*0.5 + 3*(-0.2) = 1.0 - 0.6 = 0.4
    assert pytest.approx(instantaneous_signal_phi(0.5, -0.2, 2.0, 3.0)) == 0.4


def test_compute_apgi_signal():
    # error_bias: z_i_eff = -0.2 + 0.1 = -0.1. S = 2*0.5 + 3*0.1 = 1.3
    assert (
        pytest.approx(
            compute_apgi_signal(0.5, -0.2, 2.0, 3.0, beta=0.1, dopamine_mode="error_bias")
        )
        == 1.3
    )

    # signal_additive: S = 2*0.5 + 3*0.2 + 0.1 = 1.7
    assert (
        pytest.approx(
            compute_apgi_signal(0.5, -0.2, 2.0, 3.0, beta=0.1, dopamine_mode="signal_additive")
        )
        == 1.7
    )

    with pytest.raises(ValueError, match="unknown dopamine_mode"):
        compute_apgi_signal(0.5, 0.2, 2.0, 3.0, dopamine_mode="invalid")
