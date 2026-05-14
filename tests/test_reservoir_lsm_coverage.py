import numpy as np
import pytest

from reservoir.liquid_state_machine import LiquidStateMachine


def test_lsm_init():
    lsm = LiquidStateMachine(N=20, M=2, seed=42)
    assert lsm.N == 20
    assert lsm.W_res.shape == (20, 20)

    # Error cases
    with pytest.raises(ValueError, match="spectral_radius"):
        LiquidStateMachine(spectral_radius=1.5)
    with pytest.raises(ValueError, match="N must be > 0"):
        LiquidStateMachine(N=0)
    with pytest.raises(ValueError, match="M must be > 0"):
        LiquidStateMachine(M=0)
    with pytest.raises(ValueError, match="tau_res must be > 0"):
        LiquidStateMachine(tau_res=0)
    with pytest.raises(ValueError, match="input_scale must be > 0"):
        LiquidStateMachine(input_scale=0)


def test_lsm_init_exceptions():
    from unittest.mock import patch

    # Trigger LinAlgError in init
    with patch("numpy.linalg.eigvals", side_effect=np.linalg.LinAlgError):
        lsm = LiquidStateMachine(N=10)
        assert lsm.W_res is not None


def test_lsm_step():
    lsm = LiquidStateMachine(N=10, M=2)

    # Scalar input
    x1 = lsm.step(u=0.5)
    assert x1.shape == (10,)

    # Array input
    lsm.step(u=np.array([0.1, -0.1]))

    # Scalar broadcasting
    lsm.step(u=np.array([0.7]))

    # Precision adaptive
    lsm.step(u=0.1, precision=1.0)

    # Suprathreshold
    lsm.step(u=0.1, S_target=2.0, theta=1.0, A_amp=0.5)

    # Errors
    with pytest.raises(ValueError, match="Input dimension mismatch"):
        lsm.step(u=np.array([1, 2, 3]))
    with pytest.raises(ValueError, match="tau must be > 0"):
        lsm.step(u=0.0, tau=-1.0)


def test_lsm_readout():
    lsm = LiquidStateMachine(N=5)
    lsm.x = np.ones(5)
    lsm.W_out = np.ones((5, 1))
    assert lsm.readout("linear") == 5.0
    assert lsm.readout("energy") == 5.0
    with pytest.raises(ValueError, match="Unknown readout method"):
        lsm.readout("invalid")


def test_lsm_training():
    lsm = LiquidStateMachine(N=10, M=1)
    # Collect data
    for i in range(50):
        u = float(i) / 50.0
        lsm.step(u)
        lsm.collect_state(target=u)

    X, y = lsm.get_training_data()
    assert X.shape == (50, 10)
    assert y.shape == (50,)

    res = lsm.train_readout(X, y)
    assert "rmse" in res
    assert lsm.W_out.shape == (10, 1)

    # Training errors
    with pytest.raises(ValueError, match="same number of samples"):
        lsm.train_readout(X, y[:10])
    with pytest.raises(ValueError, match="X must have 10 columns"):
        lsm.train_readout(np.zeros((10, 5)), y[:10])

    # Collection errors
    lsm.clear_history()
    with pytest.raises(ValueError, match="No training data collected"):
        lsm.get_training_data()
    lsm.collect_state(state=np.zeros(10))
    with pytest.raises(ValueError, match="No targets collected"):
        lsm.get_training_data()


def test_lsm_utility_methods():
    lsm = LiquidStateMachine(N=5)
    lsm.x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    stats = lsm.get_state_statistics()
    assert stats["mean"] == 3.0

    w_stats = lsm.get_weight_statistics()
    assert "W_res_spectral_radius" in w_stats

    lsm.reset_state()
    assert np.all(lsm.x == 0)


def test_lsm_weight_stats_exception():
    from unittest.mock import patch

    lsm = LiquidStateMachine(N=5)
    with patch("numpy.linalg.eigvals", side_effect=np.linalg.LinAlgError):
        w_stats = lsm.get_weight_statistics()
        assert w_stats["W_res_spectral_radius"] == 0.0


def test_lsm_adaptive_tau():
    lsm = LiquidStateMachine(N=5)
    # precision <= 0
    assert lsm._compute_adaptive_tau(0) == 10.0
    # precision > 0
    assert 0.1 <= lsm._compute_adaptive_tau(1.0) <= 10.0
