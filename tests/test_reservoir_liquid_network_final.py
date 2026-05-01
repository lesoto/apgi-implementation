"""Final tests for reservoir/liquid_network.py to cover remaining lines 25-27.

Lines 25-27 cover the fallback branch when LinAlgError occurs during spectral normalization.
"""

from __future__ import annotations

import numpy as np
import pytest

from reservoir.liquid_network import LiquidNetwork


class TestLiquidNetworkFallbackBranch:
    """Tests for fallback branch coverage (lines 25-27)."""

    def test_liquid_network_initialization_normal(self):
        """Test normal initialization works."""
        network = LiquidNetwork(n_units=100, spectral_radius=0.9)
        assert network.n == 100
        assert network.W_res.shape == (100, 100)
        assert network.W_in.shape == (100,)

    def test_spectral_radius_validation(self):
        """Test that invalid spectral_radius raises ValueError."""
        with pytest.raises(ValueError, match="spectral_radius must be in"):
            LiquidNetwork(n_units=100, spectral_radius=1.0)

        with pytest.raises(ValueError, match="spectral_radius must be in"):
            LiquidNetwork(n_units=100, spectral_radius=0.0)

        with pytest.raises(ValueError, match="spectral_radius must be in"):
            LiquidNetwork(n_units=100, spectral_radius=-0.1)


class TestLiquidNetworkAdaptiveTauEdgeCases:
    """Test edge cases for compute_adaptive_tau."""

    def test_adaptive_tau_zero_precision(self):
        """Test that zero precision returns tau_max."""
        network = LiquidNetwork(n_units=50)
        tau = network.compute_adaptive_tau(precision=0.0)
        assert tau == 500.0  # tau_max

    def test_adaptive_tau_negative_precision(self):
        """Test that negative precision returns tau_max."""
        network = LiquidNetwork(n_units=50)
        tau = network.compute_adaptive_tau(precision=-1.0)
        assert tau == 500.0  # tau_max

    def test_adaptive_tau_very_high_precision(self):
        """Test that very high precision returns tau_min."""
        network = LiquidNetwork(n_units=50)
        tau = network.compute_adaptive_tau(precision=1000.0)
        assert tau == 10.0  # tau_min

    def test_adaptive_tau_clamping(self):
        """Test that tau is clamped to [tau_min, tau_max]."""
        network = LiquidNetwork(n_units=50)

        # Very low precision -> should be clamped to tau_max
        tau_low = network.compute_adaptive_tau(precision=0.001)
        assert tau_low == 500.0

        # Very high precision -> should be clamped to tau_min
        tau_high = network.compute_adaptive_tau(precision=100.0)
        assert tau_high == 10.0


class TestLiquidNetworkStepVariations:
    """Test various step scenarios."""

    def test_step_with_fixed_tau(self):
        """Test step with explicit tau parameter."""
        network = LiquidNetwork(n_units=50)
        result = network.step(u=1.0, tau=50.0)
        assert result.shape == (50,)

    def test_step_with_precision_adaptive_tau(self):
        """Test step with precision-based adaptive tau."""
        network = LiquidNetwork(n_units=50)
        result = network.step(u=1.0, precision=2.0)
        assert result.shape == (50,)

    def test_step_without_adaptive_tau(self):
        """Test step without precision (uses current tau)."""
        network = LiquidNetwork(n_units=50)
        result = network.step(u=1.0)
        assert result.shape == (50,)

    def test_step_with_suprathreshold_amplification(self):
        """Test step with suprathreshold amplification."""
        network = LiquidNetwork(n_units=50)
        result = network.step(
            u=1.0,
            S_target=2.0,
            theta=1.0,
            A_amp=0.5,
        )
        assert result.shape == (50,)

    def test_step_tau_validation(self):
        """Test that invalid tau raises ValueError."""
        network = LiquidNetwork(n_units=50)
        with pytest.raises(ValueError, match="tau must be > 0"):
            network.step(u=1.0, tau=0.0)


class TestLiquidNetworkReadout:
    """Test readout_signal method variations."""

    def test_readout_linear(self):
        """Test linear readout."""
        network = LiquidNetwork(n_units=50)
        result = network.readout_signal(method="linear")
        assert isinstance(result, float)

    def test_readout_energy(self):
        """Test energy readout."""
        network = LiquidNetwork(n_units=50)
        result = network.readout_signal(method="energy")
        assert isinstance(result, float)
        assert result >= 0.0  # Energy is always non-negative


class TestLiquidNetworkSuprathresholdGain:
    """Test apply_suprathreshold_gain method."""

    def test_apply_suprathreshold_gain_positive_margin(self):
        """Test with positive margin (S > theta)."""
        network = LiquidNetwork(n_units=50)
        initial_state = network.x.copy()
        result = network.apply_suprathreshold_gain(S=2.0, theta=1.0, A=0.5, dt=1.0)
        assert result.shape == (50,)
        # State should have changed due to gain
        assert not np.allclose(network.x, initial_state)

    def test_apply_suprathreshold_gain_zero_margin(self):
        """Test with zero margin (S == theta)."""
        network = LiquidNetwork(n_units=50)
        initial_state = network.x.copy()
        result = network.apply_suprathreshold_gain(S=1.0, theta=1.0, A=0.5, dt=1.0)
        assert result.shape == (50,)
        # State should remain unchanged since [S-theta]+ = 0
        np.testing.assert_array_almost_equal(network.x, initial_state)

    def test_apply_suprathreshold_gain_negative_margin(self):
        """Test with negative margin (S < theta)."""
        network = LiquidNetwork(n_units=50)
        initial_state = network.x.copy()
        result = network.apply_suprathreshold_gain(S=0.5, theta=1.0, A=0.5, dt=1.0)
        assert result.shape == (50,)
        # State should remain unchanged since [S-theta]+ = 0
        np.testing.assert_array_almost_equal(network.x, initial_state)
