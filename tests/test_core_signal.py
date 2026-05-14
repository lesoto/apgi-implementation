"""Comprehensive unit tests for core/signal.py module.

Tests cover:
- instantaneous_signal function
- instantaneous_signal_with_dopamine function
- integrate_signal_leaky function
- stabilize_signal_log function
- compute_apgi_signal function
"""

from __future__ import annotations

import numpy as np
import pytest

from core.signal import (
    compute_apgi_signal,
    instantaneous_signal,
    instantaneous_signal_with_dopamine,
    integrate_signal_leaky,
    stabilize_signal_log,
)


class TestInstantaneousSignal:
    """Tests for instantaneous_signal function."""

    def test_basic_computation(self):
        """Should compute signal from errors and precisions."""
        result = instantaneous_signal(
            z_e=1.0,
            z_i_eff=0.5,
            pi_e_eff=2.0,
            pi_i_eff=1.0,
        )
        # S = 2.0 * |1.0| + 1.0 * |0.5| = 2.0 + 0.5 = 2.5
        assert result == 2.5

    def test_negative_errors(self):
        """Should use absolute values of errors."""
        result = instantaneous_signal(
            z_e=-1.0,
            z_i_eff=-0.5,
            pi_e_eff=2.0,
            pi_i_eff=1.0,
        )
        # S = 2.0 * |-1.0| + 1.0 * |-0.5| = 2.0 + 0.5 = 2.5
        assert result == 2.5

    def test_zero_errors(self):
        """Should handle zero errors."""
        result = instantaneous_signal(
            z_e=0.0,
            z_i_eff=0.0,
            pi_e_eff=2.0,
            pi_i_eff=1.0,
        )
        assert result == 0.0

    def test_zero_precision(self):
        """Should return zero when precisions are zero."""
        result = instantaneous_signal(
            z_e=1.0,
            z_i_eff=0.5,
            pi_e_eff=0.0,
            pi_i_eff=0.0,
        )
        assert result == 0.0


class TestInstantaneousSignalWithDopamine:
    """Tests for instantaneous_signal_with_dopamine function."""

    def test_basic_computation(self):
        """Should compute signal with dopamine term."""
        result = instantaneous_signal_with_dopamine(
            z_e=1.0,
            z_i=0.5,
            pi_e_eff=2.0,
            pi_i_eff=1.0,
            beta=0.3,
        )
        # S = 2.0 * |1.0| + 1.0 * |0.5| + 0.3 = 2.0 + 0.5 + 0.3 = 2.8
        assert result == 2.8

    def test_zero_beta(self):
        """Should match regular signal when beta is zero."""
        result = instantaneous_signal_with_dopamine(
            z_e=1.0,
            z_i=0.5,
            pi_e_eff=2.0,
            pi_i_eff=1.0,
            beta=0.0,
        )
        expected = instantaneous_signal(1.0, 0.5, 2.0, 1.0)
        assert result == expected

    def test_negative_beta(self):
        """Should handle negative dopamine bias."""
        result = instantaneous_signal_with_dopamine(
            z_e=1.0,
            z_i=0.5,
            pi_e_eff=2.0,
            pi_i_eff=1.0,
            beta=-0.3,
        )
        # S = 2.0 + 0.5 - 0.3 = 2.2
        assert result == 2.2


class TestIntegrateSignalLeaky:
    """Tests for integrate_signal_leaky function."""

    def test_basic_integration(self):
        """Should compute leaky integration correctly."""
        result = integrate_signal_leaky(
            S_prev=1.0,
            S_inst=2.0,
            lam=0.5,
        )
        # S = (1-0.5)*1.0 + 0.5*2.0 = 0.5 + 1.0 = 1.5
        assert result == 1.5

    def test_lam_zero(self):
        """Should raise ValueError for lam=0."""
        with pytest.raises(ValueError, match="lam must be in"):
            integrate_signal_leaky(1.0, 2.0, 0.0)

    def test_lam_one(self):
        """Should raise ValueError for lam=1."""
        with pytest.raises(ValueError, match="lam must be in"):
            integrate_signal_leaky(1.0, 2.0, 1.0)

    def test_lam_out_of_range(self):
        """Should raise ValueError for lam outside (0,1)."""
        with pytest.raises(ValueError, match="lam must be in"):
            integrate_signal_leaky(1.0, 2.0, -0.1)

        with pytest.raises(ValueError, match="lam must be in"):
            integrate_signal_leaky(1.0, 2.0, 1.5)

    def test_lam_half(self):
        """Should average old and new when lam=0.5."""
        result = integrate_signal_leaky(1.0, 3.0, 0.5)
        assert result == 2.0

    def test_lam_high(self):
        """Should favor new signal when lam is high."""
        result = integrate_signal_leaky(1.0, 2.0, 0.9)
        # S = 0.1*1.0 + 0.9*2.0 = 1.9
        assert result == 1.9

    def test_lam_low(self):
        """Should favor old signal when lam is low."""
        result = integrate_signal_leaky(1.0, 2.0, 0.1)
        # S = 0.9*1.0 + 0.1*2.0 = 1.1
        assert result == 1.1


class TestStabilizeSignalLog:
    """Tests for stabilize_signal_log function."""

    def test_enabled_log_stabilization(self):
        """Should apply log(1+S) when enabled."""
        result = stabilize_signal_log(1.0, enabled=True)
        expected = np.log1p(1.0)
        assert result == expected

    def test_disabled_log_stabilization(self):
        """Should return original when disabled."""
        result = stabilize_signal_log(1.0, enabled=False)
        assert result == 1.0

    def test_zero_signal(self):
        """Should handle zero signal."""
        result = stabilize_signal_log(0.0, enabled=True)
        assert result == 0.0

    def test_negative_signal(self):
        """Should handle negative signal by using max(0, S)."""
        result = stabilize_signal_log(-1.0, enabled=True)
        assert result == 0.0

    def test_large_signal(self):
        """Should handle large signals."""
        result = stabilize_signal_log(100.0, enabled=True)
        expected = np.log1p(100.0)
        assert result == expected


class TestComputeApgiSignal:
    """Tests for compute_apgi_signal function using φ(ε) = tanh(2ε) with default params."""

    def test_error_bias_mode(self):
        """Should apply φ(ε) transform with error_bias dopamine mode."""
        result = compute_apgi_signal(
            z_e=1.0,
            z_i=0.5,
            pi_e=2.0,
            pi_i_eff=1.0,
            beta=0.3,
            dopamine_mode="error_bias",
        )
        # z_i_eff = 0.5 + 0.3 = 0.8
        # S = 2.0 * φ(1.0) + 1.0 * φ(0.8)  where φ(x) = tanh(2x) at defaults
        expected = 2.0 * np.tanh(2.0 * 1.0) + 1.0 * np.tanh(2.0 * 0.8)
        assert pytest.approx(result) == expected

    def test_signal_additive_mode(self):
        """Should apply φ(ε) transform with signal_additive dopamine mode."""
        result = compute_apgi_signal(
            z_e=1.0,
            z_i=0.5,
            pi_e=2.0,
            pi_i_eff=1.0,
            beta=0.3,
            dopamine_mode="signal_additive",
        )
        # S = 2.0 * φ(1.0) + 1.0 * φ(0.5) + β
        expected = 2.0 * np.tanh(2.0 * 1.0) + 1.0 * np.tanh(2.0 * 0.5) + 0.3
        assert pytest.approx(result) == expected

    def test_invalid_mode(self):
        """Should raise ValueError for invalid mode."""
        with pytest.raises(ValueError, match="unknown dopamine_mode"):
            compute_apgi_signal(
                z_e=1.0,
                z_i=0.5,
                pi_e=2.0,
                pi_i_eff=1.0,
                beta=0.3,
                dopamine_mode="invalid_mode",
            )

    def test_zero_beta_both_modes(self):
        """Should give same result when beta=0 (z_i_eff = z_i in both modes)."""
        result_bias = compute_apgi_signal(
            z_e=1.0,
            z_i=0.5,
            pi_e=2.0,
            pi_i_eff=1.0,
            beta=0.0,
            dopamine_mode="error_bias",
        )
        result_additive = compute_apgi_signal(
            z_e=1.0,
            z_i=0.5,
            pi_e=2.0,
            pi_i_eff=1.0,
            beta=0.0,
            dopamine_mode="signal_additive",
        )
        assert pytest.approx(result_bias) == result_additive

    def test_asymmetric_valence(self):
        """Positive and negative errors should produce differently-scaled φ outputs."""
        pos = compute_apgi_signal(1.0, 1.0, 1.0, 1.0, alpha_pos=2.0, alpha_neg=0.5)
        neg = compute_apgi_signal(-1.0, -1.0, 1.0, 1.0, alpha_pos=2.0, alpha_neg=0.5)
        # With α⁺=2 > α⁻=0.5, positive errors produce larger magnitude signals
        assert abs(pos) > abs(neg)
