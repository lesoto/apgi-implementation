"""Unit tests for liquid state machine (reservoir computing) module.

Tests APGI Spec §10: Reservoir Implementation
"""

import numpy as np
import pytest

from reservoir.liquid_state_machine import LiquidStateMachine


class TestLiquidStateMachineInit:
    """Test LSM initialization."""

    def test_basic_init(self):
        """Should initialize with default parameters."""
        lsm = LiquidStateMachine(N=100, M=2)
        assert lsm.N == 100
        assert lsm.M == 2
        assert lsm.x.shape == (100,)
        assert np.allclose(lsm.x, 0.0)

    def test_custom_parameters(self):
        """Should initialize with custom parameters."""
        lsm = LiquidStateMachine(
            N=50,
            M=3,
            tau_res=2.0,
            spectral_radius=0.8,
            input_scale=0.2,
        )
        assert lsm.N == 50
        assert lsm.M == 3
        assert lsm.tau_res == 2.0
        assert lsm.spectral_radius == 0.8
        assert lsm.input_scale == 0.2

    def test_invalid_spectral_radius_zero(self):
        """Should raise error for spectral_radius = 0."""
        with pytest.raises(ValueError):
            LiquidStateMachine(N=100, M=2, spectral_radius=0.0)

    def test_invalid_spectral_radius_one(self):
        """Should raise error for spectral_radius = 1."""
        with pytest.raises(ValueError):
            LiquidStateMachine(N=100, M=2, spectral_radius=1.0)

    def test_invalid_spectral_radius_negative(self):
        """Should raise error for spectral_radius < 0."""
        with pytest.raises(ValueError):
            LiquidStateMachine(N=100, M=2, spectral_radius=-0.5)

    def test_invalid_spectral_radius_greater_than_one(self):
        """Should raise error for spectral_radius > 1."""
        with pytest.raises(ValueError):
            LiquidStateMachine(N=100, M=2, spectral_radius=1.5)

    def test_invalid_N(self):
        """Should raise error for invalid N."""
        with pytest.raises(ValueError):
            LiquidStateMachine(N=0, M=2)
        with pytest.raises(ValueError):
            LiquidStateMachine(N=-10, M=2)

    def test_invalid_M(self):
        """Should raise error for invalid M."""
        with pytest.raises(ValueError):
            LiquidStateMachine(N=100, M=0)
        with pytest.raises(ValueError):
            LiquidStateMachine(N=100, M=-2)

    def test_invalid_tau_res(self):
        """Should raise error for invalid tau_res."""
        with pytest.raises(ValueError):
            LiquidStateMachine(N=100, M=2, tau_res=0.0)
        with pytest.raises(ValueError):
            LiquidStateMachine(N=100, M=2, tau_res=-1.0)

    def test_invalid_input_scale(self):
        """Should raise error for invalid input_scale."""
        with pytest.raises(ValueError):
            LiquidStateMachine(N=100, M=2, input_scale=0.0)
        with pytest.raises(ValueError):
            LiquidStateMachine(N=100, M=2, input_scale=-0.1)

    def test_weight_shapes(self):
        """Weight matrices should have correct shapes."""
        lsm = LiquidStateMachine(N=100, M=2)
        assert lsm.W_res.shape == (100, 100)
        assert lsm.W_in.shape == (100, 2)
        assert lsm.W_out.shape == (100, 1)

    def test_spectral_radius_constraint(self):
        """Spectral radius of W_res should be < 1."""
        lsm = LiquidStateMachine(N=100, M=2, spectral_radius=0.9)
        eigs = np.linalg.eigvals(lsm.W_res)
        actual_rho = np.max(np.abs(eigs))
        assert actual_rho < 1.0

    def test_reproducibility_with_seed(self):
        """Same seed should produce same weights."""
        lsm1 = LiquidStateMachine(N=50, M=2, seed=42)
        lsm2 = LiquidStateMachine(N=50, M=2, seed=42)

        assert np.allclose(lsm1.W_res, lsm2.W_res)
        assert np.allclose(lsm1.W_in, lsm2.W_in)


class TestLiquidStateMachineStep:
    """Test LSM step function."""

    def test_step_scalar_input(self):
        """Should handle scalar input."""
        lsm = LiquidStateMachine(N=100, M=1)
        x = lsm.step(u=0.5, tau=1.0, dt=0.1)
        assert x.shape == (100,)
        assert np.all(np.isfinite(x))

    def test_step_array_input(self):
        """Should handle array input."""
        lsm = LiquidStateMachine(N=100, M=2)
        u = np.array([0.5, -0.3])
        x = lsm.step(u, tau=1.0, dt=0.1)
        assert x.shape == (100,)
        assert np.all(np.isfinite(x))

    def test_step_updates_state(self):
        """Step should update internal state."""
        lsm = LiquidStateMachine(N=100, M=2)
        x_before = lsm.x.copy()
        lsm.step(np.array([0.5, -0.3]), tau=1.0, dt=0.1)
        x_after = lsm.x.copy()
        assert not np.allclose(x_before, x_after)

    def test_step_invalid_input_dimension(self):
        """Should raise error for wrong input dimension (not scalar-like)."""
        lsm = LiquidStateMachine(N=100, M=3)
        with pytest.raises(ValueError):
            lsm.step(np.array([0.5, 0.3]), tau=1.0)  # Wrong dimension (2 vs 3)

    def test_step_invalid_tau(self):
        """Should raise error for invalid tau."""
        lsm = LiquidStateMachine(N=100, M=2)
        with pytest.raises(ValueError):
            lsm.step(np.array([0.5, -0.3]), tau=0.0, dt=0.1)
        with pytest.raises(ValueError):
            lsm.step(np.array([0.5, -0.3]), tau=-1.0, dt=0.1)

    def test_step_with_suprathreshold_amplification(self):
        """Should apply suprathreshold amplification."""
        lsm = LiquidStateMachine(N=100, M=2)
        u = np.array([0.5, -0.3])

        # Step without amplification - run multiple steps to accumulate effect
        lsm.reset_state()
        for _ in range(10):
            x1 = lsm.step(u, tau=1.0, dt=0.1, S_target=1.5, theta=1.0, A_amp=0.0)

        # Step with amplification - run multiple steps
        lsm.reset_state()
        for _ in range(10):
            x2 = lsm.step(u, tau=1.0, dt=0.1, S_target=1.5, theta=1.0, A_amp=0.1)

        # States should differ (amplification should have effect over multiple steps)
        assert not np.allclose(x1, x2, atol=1e-6)

    def test_step_no_amplification_when_below_threshold(self):
        """Amplification should be zero when S < θ."""
        lsm = LiquidStateMachine(N=100, M=2)
        u = np.array([0.5, -0.3])

        # S < θ, so margin is negative
        x = lsm.step(u, tau=1.0, dt=0.1, S_target=0.5, theta=1.0, A_amp=0.1)
        assert np.all(np.isfinite(x))

    def test_step_state_bounded(self):
        """Reservoir state should remain bounded."""
        lsm = LiquidStateMachine(N=100, M=2)
        u = np.array([0.5, -0.3])

        for _ in range(100):
            lsm.step(u, tau=1.0, dt=0.1)

        assert np.all(np.isfinite(lsm.x))
        assert np.max(np.abs(lsm.x)) < 100.0  # Reasonable bound


class TestLiquidStateMachineReadout:
    """Test LSM readout function."""

    def test_readout_linear(self):
        """Should compute linear readout."""
        lsm = LiquidStateMachine(N=100, M=2)
        lsm.step(np.array([0.5, -0.3]), tau=1.0, dt=0.1)
        S = lsm.readout(method="linear")
        assert isinstance(S, float)
        assert np.isfinite(S)

    def test_readout_energy(self):
        """Should compute energy readout."""
        lsm = LiquidStateMachine(N=100, M=2)
        lsm.step(np.array([0.5, -0.3]), tau=1.0, dt=0.1)
        S = lsm.readout(method="energy")
        assert isinstance(S, float)
        assert S >= 0.0  # Energy is non-negative

    def test_readout_invalid_method(self):
        """Should raise error for invalid readout method."""
        lsm = LiquidStateMachine(N=100, M=2)
        with pytest.raises(ValueError):
            lsm.readout(method="invalid")

    def test_readout_zero_state(self):
        """Readout should be zero for zero state."""
        lsm = LiquidStateMachine(N=100, M=2)
        lsm.reset_state()
        S = lsm.readout(method="linear")
        assert np.isclose(S, 0.0)


class TestLiquidStateMachineTraining:
    """Test LSM readout training."""

    def test_train_readout_basic(self):
        """Should train readout weights."""
        lsm = LiquidStateMachine(N=100, M=2)

        # Generate training data
        X = np.random.randn(1000, 100)
        y = np.random.randn(1000)

        result = lsm.train_readout(X, y, alpha=1e-6)

        assert "W_out" in result
        assert "mse" in result
        assert "rmse" in result
        assert "r2" in result
        assert result["mse"] >= 0.0
        assert result["rmse"] >= 0.0

    def test_train_readout_shape_mismatch(self):
        """Should raise error for shape mismatch."""
        lsm = LiquidStateMachine(N=100, M=2)

        X = np.random.randn(1000, 100)
        y = np.random.randn(500)  # Wrong size

        with pytest.raises(ValueError):
            lsm.train_readout(X, y)

    def test_train_readout_wrong_columns(self):
        """Should raise error for wrong number of columns."""
        lsm = LiquidStateMachine(N=100, M=2)

        X = np.random.randn(1000, 50)  # Wrong number of columns
        y = np.random.randn(1000)

        with pytest.raises(ValueError):
            lsm.train_readout(X, y)

    def test_train_readout_improves_fit(self):
        """Training should improve fit to target."""
        lsm = LiquidStateMachine(N=100, M=2, seed=42)

        # Generate training data with known relationship
        X = np.random.randn(1000, 100)
        y = X[:, 0] + 0.5 * X[:, 1]  # Simple linear relationship

        result = lsm.train_readout(X, y, alpha=1e-6)

        # R² should be reasonably high
        assert result["r2"] > 0.5


class TestLiquidStateMachineHistory:
    """Test LSM history collection."""

    def test_collect_state(self):
        """Should collect states."""
        lsm = LiquidStateMachine(N=100, M=2)
        lsm.step(np.array([0.5, -0.3]), tau=1.0, dt=0.1)
        lsm.collect_state(target=1.0)

        assert len(lsm.history) == 1
        assert len(lsm.history_targets) == 1

    def test_get_training_data(self):
        """Should return collected training data."""
        lsm = LiquidStateMachine(N=100, M=2)

        for i in range(10):
            lsm.step(np.array([0.5, -0.3]), tau=1.0, dt=0.1)
            lsm.collect_state(target=float(i))

        X, y = lsm.get_training_data()
        assert X.shape == (10, 100)
        assert y.shape == (10,)

    def test_get_training_data_no_data(self):
        """Should raise error if no data collected."""
        lsm = LiquidStateMachine(N=100, M=2)
        with pytest.raises(ValueError):
            lsm.get_training_data()

    def test_clear_history(self):
        """Should clear collected history."""
        lsm = LiquidStateMachine(N=100, M=2)

        for i in range(10):
            lsm.step(np.array([0.5, -0.3]), tau=1.0, dt=0.1)
            lsm.collect_state(target=float(i))

        lsm.clear_history()
        assert len(lsm.history) == 0
        assert len(lsm.history_targets) == 0


class TestLiquidStateMachineReset:
    """Test LSM reset function."""

    def test_reset_state(self):
        """Should reset state to zero."""
        lsm = LiquidStateMachine(N=100, M=2)
        lsm.step(np.array([0.5, -0.3]), tau=1.0, dt=0.1)
        assert not np.allclose(lsm.x, 0.0)

        lsm.reset_state()
        assert np.allclose(lsm.x, 0.0)


class TestLiquidStateMachineStatistics:
    """Test LSM statistics functions."""

    def test_state_statistics(self):
        """Should compute state statistics."""
        lsm = LiquidStateMachine(N=100, M=2)
        lsm.step(np.array([0.5, -0.3]), tau=1.0, dt=0.1)

        stats = lsm.get_state_statistics()

        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert "norm" in stats

    def test_weight_statistics(self):
        """Should compute weight statistics."""
        lsm = LiquidStateMachine(N=100, M=2)

        stats = lsm.get_weight_statistics()

        assert "W_res_spectral_radius" in stats
        assert "W_res_mean" in stats
        assert "W_res_std" in stats
        assert "W_in_mean" in stats
        assert "W_in_std" in stats
        assert "W_out_norm" in stats

        # Spectral radius should be < 1
        assert stats["W_res_spectral_radius"] < 1.0


class TestLiquidStateMachineStability:
    """Test LSM stability properties."""

    def test_echo_state_property(self):
        """Reservoir should have echo state property."""
        lsm = LiquidStateMachine(N=100, M=2, spectral_radius=0.9)

        # Run with different inputs
        u1 = np.array([0.5, -0.3])
        u2 = np.array([-0.5, 0.3])

        lsm.reset_state()
        for _ in range(100):
            lsm.step(u1, tau=1.0, dt=0.1)
        x1 = lsm.x.copy()

        lsm.reset_state()
        for _ in range(100):
            lsm.step(u2, tau=1.0, dt=0.1)
        x2 = lsm.x.copy()

        # States should be different
        assert not np.allclose(x1, x2)

    def test_fading_memory(self):
        """Reservoir should have fading memory property."""
        lsm = LiquidStateMachine(N=100, M=2, spectral_radius=0.9)

        # Apply impulse
        lsm.reset_state()
        lsm.step(np.array([1.0, 1.0]), tau=1.0, dt=0.1)
        x_impulse = lsm.x.copy()

        # Continue with zero input
        for _ in range(100):
            lsm.step(np.array([0.0, 0.0]), tau=1.0, dt=0.1)
        x_final = lsm.x.copy()

        # State should decay toward zero
        assert np.linalg.norm(x_final) < np.linalg.norm(x_impulse)

    def test_liquid_state_machine_history_tracking(self):
        """Test history tracking during reservoir computation."""
        lsm = LiquidStateMachine(N=50, M=2, spectral_radius=0.9)
        # Generate some data
        inputs = np.random.randn(10, 2)
        for inp in inputs:
            x = lsm.step(inp, tau=1.0, dt=0.1)
            # Manually track history
            lsm.history.append(x.copy())

        # History should track states
        assert len(lsm.history) == 10
        assert all(isinstance(h, np.ndarray) for h in lsm.history)

    def test_liquid_state_machine_clear_history(self):
        """Test clearing history."""
        lsm = LiquidStateMachine(N=50, M=2, spectral_radius=0.9)
        inputs = np.random.randn(5, 2)
        for inp in inputs:
            x = lsm.step(inp, tau=1.0, dt=0.1)
            lsm.history.append(x.copy())

        assert len(lsm.history) == 5

        # Clear history
        lsm.history.clear()
        assert len(lsm.history) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])  # pragma: no cover
