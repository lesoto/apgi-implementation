"""Extended tests for reservoir/liquid_network.py to achieve 100% coverage."""

import numpy as np
import pytest
from reservoir.liquid_network import LiquidNetwork


class TestLiquidNetworkExtended:
    """Extended tests for LiquidNetwork class."""

    def test_step_with_tau(self):
        """Should use provided tau parameter."""
        network = LiquidNetwork(n_units=50)
        u = 1.0
        result = network.step(u, tau=50.0, dt=1.0)
        assert len(result) == 50

    def test_step_with_precision_adaptive(self):
        """Should compute adaptive tau from precision."""
        network = LiquidNetwork(n_units=50)
        u = 1.0
        result = network.step(u, precision=2.0, dt=1.0)
        assert len(result) == 50
        # tau_current should have been updated
        assert network.tau_current != 100.0  # Default was changed

    def test_step_tau_validation(self):
        """Should raise ValueError for non-positive tau."""
        network = LiquidNetwork(n_units=50)
        with pytest.raises(ValueError, match="tau must be > 0"):
            network.step(1.0, tau=0.0, dt=1.0)

    def test_readout_energy(self):
        """Should compute energy readout."""
        network = LiquidNetwork(n_units=50)
        network.step(1.0, tau=100.0, dt=1.0)
        energy = network.readout_signal(method="energy")
        assert isinstance(energy, float)
        assert energy >= 0  # Energy is always non-negative

    def test_readout_invalid_method(self):
        """Should raise ValueError for invalid method."""
        network = LiquidNetwork(n_units=50)
        with pytest.raises(ValueError, match="Unknown readout method"):
            network.readout_signal(method="invalid")

    def test_apply_suprathreshold_gain(self):
        """Should apply suprathreshold gain."""
        network = LiquidNetwork(n_units=50)
        network.step(1.0, tau=100.0, dt=1.0)
        original_x = network.x.copy()
        network.apply_suprathreshold_gain(S=1.5, theta=1.0, A=0.1, dt=1.0)
        # x should have changed
        assert not np.array_equal(network.x, original_x)

    def test_compute_adaptive_tau_zero_precision(self):
        """Should return tau_max when precision is zero or negative."""
        network = LiquidNetwork(n_units=50)
        tau = network.compute_adaptive_tau(precision=0.0, tau_min=10.0, tau_max=500.0)
        assert tau == 500.0

        tau = network.compute_adaptive_tau(precision=-1.0, tau_min=10.0, tau_max=500.0)
        assert tau == 500.0

    def test_step_without_adaptive_tau(self):
        """Should use current tau when neither tau nor precision provided."""
        network = LiquidNetwork(n_units=50)
        network.tau_current = 50.0
        u = 1.0
        result = network.step(u, dt=1.0)  # No tau or precision
        assert len(result) == 50
        # tau_current should remain unchanged
        assert network.tau_current == 50.0

    def test_step_with_suprathreshold_gain(self):
        """Should apply suprathreshold gain when S > theta."""
        network = LiquidNetwork(n_units=50)
        network.step(1.0, tau=100.0, dt=1.0)
        original_x = network.x.copy()
        # Apply suprathreshold gain with S > theta
        network.step(
            1.0,
            tau=100.0,
            dt=1.0,
            S_target=1.5,
            theta=1.0,
            A_amp=0.1,
        )
        # x should have changed due to suprathreshold gain
        assert not np.array_equal(network.x, original_x)
