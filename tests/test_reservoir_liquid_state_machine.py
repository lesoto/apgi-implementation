"""Comprehensive unit tests for reservoir/liquid_state_machine.py module.

Tests cover:
- LiquidStateMachine class
"""

from __future__ import annotations

import numpy as np
import pytest

from reservoir.liquid_state_machine import LiquidStateMachine


class TestLiquidStateMachine:
    """Tests for LiquidStateMachine class."""

    def test_initialization(self):
        """Should initialize correctly."""
        lsm = LiquidStateMachine(N=50, M=2)
        assert lsm.N == 50
        assert lsm.M == 2
        assert len(lsm.x) == 50
        assert lsm.W_res.shape == (50, 50)
        assert lsm.W_in.shape == (50, 2)

    def test_invalid_spectral_radius(self):
        """Should raise ValueError for invalid spectral radius."""
        with pytest.raises(ValueError, match="spectral_radius must be in"):
            LiquidStateMachine(N=50, spectral_radius=1.5)

        with pytest.raises(ValueError, match="spectral_radius must be in"):
            LiquidStateMachine(N=50, spectral_radius=0.0)

    def test_invalid_dimensions(self):
        """Should raise ValueError for invalid dimensions."""
        with pytest.raises(ValueError, match="N must be > 0"):
            LiquidStateMachine(N=0)

        with pytest.raises(ValueError, match="M must be > 0"):
            LiquidStateMachine(N=50, M=0)

    def test_step(self):
        """Should update reservoir state."""
        lsm = LiquidStateMachine(N=50, M=2)
        initial_state = lsm.x.copy()

        u = np.array([0.5, -0.3])
        new_state = lsm.step(u, tau=1.0, dt=0.1)

        assert len(new_state) == 50
        assert not np.allclose(new_state, initial_state)

    def test_step_scalar_input(self):
        """Should handle scalar input."""
        lsm = LiquidStateMachine(N=50, M=1)
        result = lsm.step(0.5, tau=1.0, dt=0.1)
        assert len(result) == 50

    def test_readout_linear(self):
        """Should compute linear readout."""
        lsm = LiquidStateMachine(N=50, M=2)
        lsm.step(np.array([0.5, -0.3]), tau=1.0)

        result = lsm.readout(method="linear")
        assert isinstance(result, float)

    def test_readout_energy(self):
        """Should compute energy readout."""
        lsm = LiquidStateMachine(N=50, M=2)
        lsm.step(np.array([0.5, -0.3]), tau=1.0)

        result = lsm.readout(method="energy")
        assert result >= 0  # Energy is non-negative

    def test_invalid_readout_method(self):
        """Should raise ValueError for invalid readout method."""
        lsm = LiquidStateMachine(N=50, M=2)
        with pytest.raises(ValueError, match="Unknown readout method"):
            lsm.readout(method="invalid")

    def test_train_readout(self):
        """Should train readout weights."""
        lsm = LiquidStateMachine(N=50, M=2)

        # Generate training data
        X = np.random.randn(100, 50)
        y = np.random.randn(100)

        result = lsm.train_readout(X, y, alpha=1e-6)

        assert "W_out" in result
        assert "mse" in result
        assert "rmse" in result
        assert "r2" in result

    def test_invalid_training_shapes(self):
        """Should raise ValueError for invalid shapes."""
        lsm = LiquidStateMachine(N=50, M=2)

        with pytest.raises(ValueError):
            X = np.random.randn(100, 50)
            y = np.random.randn(50)  # Mismatched length
            lsm.train_readout(X, y)

    def test_collect_state(self):
        """Should collect states for training."""
        lsm = LiquidStateMachine(N=50, M=2)

        lsm.collect_state(target=1.0)
        lsm.collect_state(target=2.0)

        assert len(lsm.history) == 2
        assert len(lsm.history_targets) == 2

    def test_get_training_data(self):
        """Should return collected training data."""
        lsm = LiquidStateMachine(N=50, M=2)

        for i in range(10):
            lsm.step(np.array([0.5, -0.3]), tau=1.0)
            lsm.collect_state(target=float(i))

        X, y = lsm.get_training_data()
        assert X.shape == (10, 50)
        assert y.shape == (10,)

    def test_get_training_data_empty(self):
        """Should raise ValueError for empty data."""
        lsm = LiquidStateMachine(N=50, M=2)

        with pytest.raises(ValueError, match="No training data collected"):
            lsm.get_training_data()

    def test_clear_history(self):
        """Should clear collected data."""
        lsm = LiquidStateMachine(N=50, M=2)

        for _ in range(5):
            lsm.collect_state(target=1.0)

        lsm.clear_history()
        assert len(lsm.history) == 0
        assert len(lsm.history_targets) == 0

    def test_reset_state(self):
        """Should reset state to zero."""
        lsm = LiquidStateMachine(N=50, M=2)

        for _ in range(5):
            lsm.step(np.array([0.5, -0.3]), tau=1.0)

        lsm.reset_state()
        assert np.allclose(lsm.x, np.zeros(50))

    def test_get_state_statistics(self):
        """Should return state statistics."""
        lsm = LiquidStateMachine(N=50, M=2)
        lsm.step(np.array([0.5, -0.3]), tau=1.0)

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
        assert "W_res_std" in stats
        assert "W_in_mean" in stats
        assert "W_in_std" in stats
        assert "W_out_norm" in stats

    def test_input_scale_attribute(self):
        """Should have input_scale attribute."""
        lsm = LiquidStateMachine(N=50, M=2, input_scale=0.5)
        assert lsm.input_scale == 0.5

    def test_step_with_precision_modulation(self):
        """Should handle precision-modulated timescale."""
        lsm = LiquidStateMachine(N=50, M=2)
        u = np.array([0.5, -0.3])

        # Step with different tau (precision modulation)
        result1 = lsm.step(u, tau=0.5, dt=0.1)
        result2 = lsm.step(u, tau=2.0, dt=0.1)

        # Results should differ due to different timescales
        assert not np.allclose(result1, result2)

    def test_suprathreshold_amplification(self):
        """Should handle suprathreshold amplification."""
        lsm = LiquidStateMachine(N=50, M=2)

        # Step with high input to trigger amplification
        u = np.array([5.0, 5.0])
        result = lsm.step(u, tau=1.0, dt=0.1)

        # State should be significantly affected
        assert np.linalg.norm(result) > 0.1

    def test_train_readout_with_regularization(self):
        """Should train readout with regularization."""
        lsm = LiquidStateMachine(N=50, M=2)

        X = np.random.randn(100, 50)
        y = np.random.randn(100)

        result = lsm.train_readout(X, y, alpha=1e-6)

        assert "W_out" in result
        assert "mse" in result
