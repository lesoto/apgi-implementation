import numpy as np
import pytest

from reservoir.liquid_network import LiquidNetwork


def test_liquid_network_init():
    net = LiquidNetwork(n_units=10, spectral_radius=0.8)  # §17: must be in [0.7, 0.95]
    assert net.n == 10
    assert net.W_res.shape == (10, 10)
    assert net.W_in.shape == (10,)
    assert net.W_out.shape == (10,)

    with pytest.raises(ValueError, match="spectral_radius"):
        LiquidNetwork(spectral_radius=1.5)


def test_liquid_network_init_exceptions():
    from unittest.mock import patch

    # Trigger LinAlgError in eigvals
    with patch("numpy.linalg.eigvals", side_effect=np.linalg.LinAlgError):
        net = LiquidNetwork(n_units=5, spectral_radius=0.8)
        assert net.W_res is not None


def test_compute_adaptive_tau():
    net = LiquidNetwork(n_units=5)
    # precision <= 0
    assert net.compute_adaptive_tau(0) == 500.0
    assert net.compute_adaptive_tau(-1) == 500.0

    # Normal case
    tau = net.compute_adaptive_tau(1.0)
    assert 10.0 <= tau <= 500.0

    # Clipping
    assert net.compute_adaptive_tau(1e6) == 10.0
    assert net.compute_adaptive_tau(1e-6) == 500.0


def test_liquid_network_step():
    net = LiquidNetwork(n_units=10)
    x0 = net.x.copy()

    # Step with fixed tau
    x1 = net.step(u=1.0, tau=50.0)
    assert not np.array_equal(x0, x1)

    # Step with precision
    net.step(u=0.5, precision=2.0)
    assert net.tau_current != 100.0

    # Step with default tau (previous current)
    net.step(u=0.0)

    # Step with suprathreshold amplification
    net.step(u=0.0, S_target=1.0, theta=0.5, A_amp=1.0)

    # Error: tau <= 0
    with pytest.raises(ValueError, match="tau must be > 0"):
        net.step(u=0.0, tau=0.0)


def test_readout_signal():
    net = LiquidNetwork(n_units=5)
    net.x = np.ones(5)
    assert isinstance(net.readout_signal("linear"), float)
    assert isinstance(net.readout_signal("energy"), float)

    with pytest.raises(ValueError, match="Unknown readout method"):
        net.readout_signal("invalid")


def test_apply_suprathreshold_gain():
    net = LiquidNetwork(n_units=5)
    x0 = net.x.copy()
    net.apply_suprathreshold_gain(S=1.0, theta=0.5, A=1.0)
    assert not np.array_equal(x0, net.x)

    # Below threshold
    x_pre = net.x.copy()
    net.apply_suprathreshold_gain(S=0.1, theta=0.5)
    assert np.array_equal(x_pre, net.x)
