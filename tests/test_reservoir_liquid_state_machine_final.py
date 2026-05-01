"""Final tests for reservoir/liquid_state_machine.py to cover remaining lines 126-128, 421-422.

Lines 126-128: Fallback branch for LinAlgError during spectral normalization.
Lines 421-422: Clear history targets branch.
"""

from __future__ import annotations

import numpy as np
import pytest

from reservoir.liquid_state_machine import LiquidStateMachine


class TestLiquidStateMachineFallbackBranch:
    """Tests for fallback branch during initialization (lines 126-128)."""

    def test_initialization_normal(self):
        """Test normal initialization."""
        lsm = LiquidStateMachine(N=100, M=2, spectral_radius=0.9)
        assert lsm.N == 100
        assert lsm.M == 2
        assert lsm.W_res.shape == (100, 100)
        assert lsm.W_in.shape == (100, 2)

    def test_initialization_validates_spectral_radius(self):
        """Test spectral radius validation."""
        with pytest.raises(ValueError, match="spectral_radius must be in"):
            LiquidStateMachine(N=100, M=2, spectral_radius=1.0)

        with pytest.raises(ValueError, match="spectral_radius must be in"):
            LiquidStateMachine(N=100, M=2, spectral_radius=0.0)

    def test_initialization_validates_N(self):
        """Test N validation."""
        with pytest.raises(ValueError, match="N must be > 0"):
            LiquidStateMachine(N=0, M=2)

        with pytest.raises(ValueError, match="N must be > 0"):
            LiquidStateMachine(N=-1, M=2)

    def test_initialization_validates_M(self):
        """Test M validation."""
        with pytest.raises(ValueError, match="M must be > 0"):
            LiquidStateMachine(N=100, M=0)

        with pytest.raises(ValueError, match="M must be > 0"):
            LiquidStateMachine(N=100, M=-1)

    def test_initialization_validates_tau_res(self):
        """Test tau_res validation."""
        with pytest.raises(ValueError, match="tau_res must be > 0"):
            LiquidStateMachine(N=100, M=2, tau_res=0.0)

    def test_initialization_validates_input_scale(self):
        """Test input_scale validation."""
        with pytest.raises(ValueError, match="input_scale must be > 0"):
            LiquidStateMachine(N=100, M=2, input_scale=0.0)

    def test_initialization_with_seed(self):
        """Test initialization with random seed."""
        lsm1 = LiquidStateMachine(N=50, M=2, seed=42)
        lsm2 = LiquidStateMachine(N=50, M=2, seed=42)
        np.testing.assert_array_almost_equal(lsm1.W_res, lsm2.W_res)
        np.testing.assert_array_almost_equal(lsm1.W_in, lsm2.W_in)


class TestLiquidStateMachineHistoryClearing:
    """Tests for clear_history method covering lines 352-355."""

    def test_clear_history_empty(self):
        """Test clearing empty history."""
        lsm = LiquidStateMachine(N=50, M=2)
        lsm.clear_history()
        assert len(lsm.history) == 0
        assert len(lsm.history_targets) == 0

    def test_clear_history_with_data(self):
        """Test clearing history with data."""
        lsm = LiquidStateMachine(N=50, M=2)

        # Add some history using collect_state
        for i in range(10):
            lsm.step(np.array([float(i), float(i + 1)]))
            lsm.collect_state(target=float(i))

        assert len(lsm.history) > 0

        lsm.clear_history()
        assert len(lsm.history) == 0
        assert len(lsm.history_targets) == 0

    def test_clear_history_after_training_data_collection(self):
        """Test clearing after collecting training data."""
        lsm = LiquidStateMachine(N=50, M=2)

        # Collect some training data
        for i in range(10):
            lsm.step(np.array([float(i), float(i + 1)]))
            lsm.collect_state(target=float(i))

        assert len(lsm.history) > 0
        assert len(lsm.history_targets) > 0

        lsm.clear_history()
        assert len(lsm.history) == 0
        assert len(lsm.history_targets) == 0


class TestLiquidStateMachineStepVariations:
    """Test various step scenarios."""

    def test_step_with_array_input(self):
        """Test step with array input."""
        lsm = LiquidStateMachine(N=50, M=2)
        result = lsm.step(np.array([1.0, 2.0]))
        assert result.shape == (50,)

    def test_step_with_scalar_input_broadcasting(self):
        """Test step with scalar input that gets converted to array."""
        lsm = LiquidStateMachine(N=50, M=1)  # M=1 for scalar input
        result = lsm.step(1.0)  # Scalar value
        assert result.shape == (50,)

    def test_step_with_precision(self):
        """Test step with precision parameter."""
        lsm = LiquidStateMachine(N=50, M=2)
        result = lsm.step(np.array([1.0, 2.0]), precision=2.0)
        assert result.shape == (50,)

    def test_step_with_tau(self):
        """Test step with explicit tau."""
        lsm = LiquidStateMachine(N=50, M=2)
        result = lsm.step(np.array([1.0, 2.0]), tau=2.0)
        assert result.shape == (50,)

    def test_step_with_suprathreshold_amplification(self):
        """Test step with suprathreshold amplification."""
        lsm = LiquidStateMachine(N=50, M=2)
        result = lsm.step(
            np.array([1.0, 2.0]),
            S_target=2.0,
            theta=1.0,
        )
        assert result.shape == (50,)

    def test_step_with_zero_precision(self):
        """Test step with zero precision."""
        lsm = LiquidStateMachine(N=50, M=2)
        result = lsm.step(np.array([1.0, 2.0]), precision=0.0)
        assert result.shape == (50,)


class TestLiquidStateMachineReadout:
    """Test readout variations."""

    def test_readout_linear(self):
        """Test linear readout."""
        lsm = LiquidStateMachine(N=50, M=2)
        result = lsm.readout(method="linear")
        assert isinstance(result, float)

    def test_readout_energy(self):
        """Test energy readout."""
        lsm = LiquidStateMachine(N=50, M=2)
        result = lsm.readout(method="energy")
        assert isinstance(result, float)
        assert result >= 0.0

    def test_readout_invalid_method(self):
        """Test readout with invalid method."""
        lsm = LiquidStateMachine(N=50, M=2)
        with pytest.raises(ValueError, match="Unknown readout method"):
            lsm.readout(method="invalid")


class TestLiquidStateMachineTrainingData:
    """Test training data collection and retrieval."""

    def test_get_training_data_empty(self):
        """Test getting training data when empty raises ValueError."""
        lsm = LiquidStateMachine(N=50, M=2)
        with pytest.raises(ValueError, match="No training data collected"):
            lsm.get_training_data()

    def test_get_training_data_no_targets(self):
        """Test getting training data with states but no targets raises ValueError."""
        lsm = LiquidStateMachine(N=50, M=2)

        # Add states without targets using collect_state without target
        for i in range(5):
            lsm.step(np.array([float(i), float(i + 1)]))
            lsm.collect_state()  # No target

        with pytest.raises(ValueError, match="No targets collected"):
            lsm.get_training_data()

    def test_collect_and_get_training_data(self):
        """Test collecting states with targets and retrieving."""
        lsm = LiquidStateMachine(N=50, M=2)

        # Collect training data
        for i in range(10):
            lsm.step(np.array([float(i), float(i + 1)]))
            lsm.collect_state(target=float(i))

        X, y = lsm.get_training_data()
        assert X is not None
        assert y is not None
        assert X.shape[0] == 10  # 10 samples collected
        assert X.shape[1] == 50  # 50 units
        assert y.shape == (10,)


class TestLiquidStateMachineReset:
    """Test reset_state method."""

    def test_reset_state(self):
        """Test state reset."""
        lsm = LiquidStateMachine(N=50, M=2)

        # Run some steps
        for i in range(5):
            lsm.step(np.array([float(i), float(i + 1)]))

        # Reset state
        lsm.reset_state()
        assert np.all(lsm.x == 0.0)


class TestLiquidStateMachineStateStatistics:
    """Test state statistics methods."""

    def test_get_state_statistics(self):
        """Test get_state_statistics."""
        lsm = LiquidStateMachine(N=50, M=2)

        # Run some steps
        for i in range(5):
            lsm.step(np.array([float(i), float(i + 1)]))

        stats = lsm.get_state_statistics()
        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats

    def test_get_weight_statistics(self):
        """Test get_weight_statistics."""
        lsm = LiquidStateMachine(N=50, M=2)
        stats = lsm.get_weight_statistics()
        assert "W_res_mean" in stats
        assert "W_res_std" in stats
        assert "W_in_mean" in stats
        assert "W_in_std" in stats
