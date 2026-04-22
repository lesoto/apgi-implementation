"""Comprehensive unit tests for reservoir/liquid_network.py module.

Tests cover:
- LiquidNetwork class
"""

from __future__ import annotations

import numpy as np
import pytest

from reservoir.liquid_network import LiquidNetwork


class TestLiquidNetwork:
    """Tests for LiquidNetwork class."""

    def test_initialization(self):
        """Should initialize correctly."""
        network = LiquidNetwork(n_units=100, spectral_radius=0.9)
        assert network.n == 100
        assert len(network.x) == 100
        assert network.W_res.shape == (100, 100)

    def test_invalid_spectral_radius(self):
        """Should raise ValueError for invalid spectral radius."""
        with pytest.raises(ValueError, match="spectral_radius must be in"):
            LiquidNetwork(n_units=100, spectral_radius=1.0)

    def test_compute_adaptive_tau(self):
        """Should compute adaptive time constant."""
        network = LiquidNetwork(n_units=100)

        result = network.compute_adaptive_tau(precision=1.0)
        assert result > 0

    def test_adaptive_tau_inverse_relation(self):
        """High precision should give low tau."""
        network = LiquidNetwork(n_units=100)

        tau_low_pi = network.compute_adaptive_tau(precision=0.1)
        tau_high_pi = network.compute_adaptive_tau(precision=10.0)

        assert tau_high_pi < tau_low_pi

    def test_adaptive_tau_clamping(self):
        """Should clamp to min and max."""
        network = LiquidNetwork(n_units=100)

        tau_low = network.compute_adaptive_tau(
            precision=1e6, tau_min=10.0, tau_max=500.0
        )
        tau_high = network.compute_adaptive_tau(
            precision=1e-6, tau_min=10.0, tau_max=500.0
        )

        assert tau_low >= 10.0
        assert tau_high <= 500.0

    def test_step(self):
        """Should update network state."""
        network = LiquidNetwork(n_units=100)
        initial_state = network.x.copy()

        result = network.step(u=0.5, tau=100.0, dt=1.0)

        assert len(result) == 100
        assert not np.allclose(result, initial_state)

    def test_step_with_adaptive_tau(self):
        """Should use adaptive tau when precision provided."""
        network = LiquidNetwork(n_units=100)

        result = network.step(u=0.5, precision=1.0, dt=1.0)
        assert len(result) == 100

    def test_readout_signal(self):
        """Should compute readout signal."""
        network = LiquidNetwork(n_units=100)
        network.step(u=0.5, tau=100.0)

        result = network.readout_signal(method="linear")
        assert isinstance(result, float)

        result = network.readout_signal(method="energy")
        assert result >= 0  # Energy is non-negative

    def test_invalid_readout_method(self):
        """Should raise ValueError for invalid method."""
        network = LiquidNetwork(n_units=100)
        with pytest.raises(ValueError, match="Unknown readout method"):
            network.readout_signal(method="invalid")

    def test_apply_suprathreshold_gain(self):
        """Should apply suprathreshold gain."""
        network = LiquidNetwork(n_units=100)
        network.step(u=0.5, tau=100.0)

        initial_state = network.x.copy()
        network.apply_suprathreshold_gain(S=1.5, theta=1.0, A=1.0, dt=1.0)

        assert not np.allclose(network.x, initial_state)
