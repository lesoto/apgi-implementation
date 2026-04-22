"""Extended tests for reservoir/liquid_state_machine.py to achieve 100% coverage."""

import numpy as np
import pytest
from reservoir.liquid_state_machine import LiquidStateMachine


class TestLiquidStateMachineExtended:
    """Extended tests for LiquidStateMachine class."""

    def test_step_with_invalid_input_dimension(self):
        """Should raise ValueError for mismatched input dimension."""
        lsm = LiquidStateMachine(N=50, M=3)  # Expects 3D input
        with pytest.raises(ValueError, match="Input dimension mismatch"):
            lsm.step(np.array([1.0, 2.0]), tau=10.0)  # Only 2D input

    def test_step_with_tau(self):
        """Should use provided tau."""
        lsm = LiquidStateMachine(N=50, M=2)
        u = np.array([1.0, 0.5])
        result = lsm.step(u, tau=50.0, dt=1.0)
        assert len(result) == 50

    def test_step_with_precision(self):
        """Should compute adaptive tau from precision."""
        lsm = LiquidStateMachine(N=50, M=2, tau_res=100.0)
        u = np.array([1.0, 0.5])
        result = lsm.step(u, precision=2.0, dt=1.0)
        assert len(result) == 50

    def test_step_with_default_tau(self):
        """Should use default tau_res when neither tau nor precision provided."""
        lsm = LiquidStateMachine(N=50, M=2, tau_res=100.0)
        u = np.array([1.0, 0.5])
        result = lsm.step(u, dt=1.0)  # No tau or precision
        assert len(result) == 50

    def test_step_tau_validation(self):
        """Should raise ValueError for non-positive tau."""
        lsm = LiquidStateMachine(N=50, M=2)
        with pytest.raises(ValueError, match="tau must be > 0"):
            lsm.step(np.array([1.0, 0.5]), tau=0.0, dt=1.0)

    def test_readout_energy(self):
        """Should compute energy readout."""
        lsm = LiquidStateMachine(N=50, M=2)
        lsm.step(np.array([1.0, 0.5]), tau=10.0, dt=1.0)
        energy = lsm.readout(method="energy")
        assert isinstance(energy, float)
        assert energy >= 0

    def test_readout_invalid_method(self):
        """Should raise ValueError for invalid method."""
        lsm = LiquidStateMachine(N=50, M=2)
        with pytest.raises(ValueError, match="Unknown readout method"):
            lsm.readout(method="invalid")

    def test_get_training_data_empty(self):
        """Should raise ValueError when no data collected."""
        lsm = LiquidStateMachine(N=50, M=2)
        with pytest.raises(ValueError, match="No training data collected"):
            lsm.get_training_data()

    def test_get_training_data_no_targets(self):
        """Should raise ValueError when no targets collected."""
        lsm = LiquidStateMachine(N=50, M=2)
        # Manually add history without targets
        lsm.history.append(lsm.x.copy())
        with pytest.raises(ValueError, match="No targets collected"):
            lsm.get_training_data()

    def test_collect_and_get_training_data(self):
        """Should collect and return training data."""
        lsm = LiquidStateMachine(N=50, M=2)
        u = np.array([1.0, 0.5])
        for i in range(5):
            lsm.step(u, tau=10.0, dt=1.0)
            lsm.collect_state(target=float(i))

        X, y = lsm.get_training_data()
        assert X.shape == (5, 50)
        assert y.shape == (5,)

    def test_clear_history(self):
        """Should clear collected data."""
        lsm = LiquidStateMachine(N=50, M=2)
        u = np.array([1.0, 0.5])
        lsm.step(u, tau=10.0, dt=1.0)
        lsm.collect_state(target=1.0)

        assert len(lsm.history) > 0
        lsm.clear_history()
        assert len(lsm.history) == 0
        assert len(lsm.history_targets) == 0

    def test_reset_state(self):
        """Should reset state to zero."""
        lsm = LiquidStateMachine(N=50, M=2)
        u = np.array([1.0, 0.5])
        lsm.step(u, tau=10.0, dt=1.0)

        assert not np.allclose(lsm.x, 0)
        lsm.reset_state()
        assert np.allclose(lsm.x, 0)

    def test_compute_adaptive_tau_zero_precision(self):
        """Should return tau_max when precision is zero or negative."""
        lsm = LiquidStateMachine(N=50, M=2, tau_res=100.0)
        tau = lsm._compute_adaptive_tau(precision=0.0, tau_min=0.1, tau_max=10.0)
        assert tau == 10.0

        tau = lsm._compute_adaptive_tau(precision=-1.0, tau_min=0.1, tau_max=10.0)
        assert tau == 10.0

    def test_get_state_statistics(self):
        """Should return state statistics."""
        lsm = LiquidStateMachine(N=50, M=2)
        stats = lsm.get_state_statistics()
        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert "norm" in stats

    def test_get_weight_statistics(self):
        """Should return weight statistics."""
        lsm = LiquidStateMachine(N=50, M=2)
        stats = lsm.get_weight_statistics()
        assert "W_res_spectral_radius" in stats
        assert "W_res_mean" in stats
        assert "W_in_mean" in stats
        assert "W_out_norm" in stats
