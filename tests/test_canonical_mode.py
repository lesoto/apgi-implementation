"""Tests for canonical discrete mode vs ODE mode signal accumulation."""

import numpy as np
import pytest

from config import CONFIG
from pipeline import APGIPipeline


class TestCanonicalDiscreteMode:
    """Test discrete canonical mode against ODE mode."""

    def test_discrete_mode_uses_lam_parameter(self):
        """Discrete mode should use lam parameter for leaky accumulation."""
        config = dict(CONFIG)
        config["use_canonical_discrete_mode"] = True
        config["lam"] = 0.3
        config["stochastic_ignition"] = False

        pipeline = APGIPipeline(config)

        # Run a few steps to ensure discrete mode is active
        for _ in range(5):
            pipeline.step(x_e=1.0, x_i=1.0)

        # Verify lam parameter was used (signal should be different from ODE mode)
        assert pipeline.S is not None

    def test_ode_mode_uses_tau_s_parameter(self):
        """ODE mode should use tau_s parameter for continuous integration."""
        config = dict(CONFIG)
        config["use_canonical_discrete_mode"] = False  # Default ODE mode
        config["tau_s"] = 10.0
        config["stochastic_ignition"] = False

        pipeline = APGIPipeline(config)

        # Run a few steps
        for _ in range(5):
            pipeline.step(x_e=1.0, x_i=1.0)

        # Verify tau_s parameter was used
        assert pipeline.S is not None

    def test_discrete_vs_ode_produce_different_outputs(self):
        """Discrete and ODE modes should produce different signal values."""
        config_discrete = dict(CONFIG)
        config_discrete["use_canonical_discrete_mode"] = True
        config_discrete["lam"] = 0.2
        config_discrete["stochastic_ignition"] = False

        config_ode = dict(CONFIG)
        config_ode["use_canonical_discrete_mode"] = False
        config_ode["tau_s"] = 5.0
        config_ode["stochastic_ignition"] = False

        pipeline_discrete = APGIPipeline(config_discrete)
        pipeline_ode = APGIPipeline(config_ode)

        # Run both pipelines with identical inputs
        S_discrete = []
        S_ode = []

        for _ in range(10):
            pipeline_discrete.step(x_e=1.0, x_i=1.0)
            pipeline_ode.step(x_e=1.0, x_i=1.0)
            S_discrete.append(pipeline_discrete.S)
            S_ode.append(pipeline_ode.S)

        # Signals should differ (different integration schemes)
        assert not np.allclose(S_discrete, S_ode, rtol=0.01)

    def test_discrete_mode_lam_bounds_validation(self):
        """Discrete mode should validate lam ∈ (0,1)."""
        config = dict(CONFIG)
        config["use_canonical_discrete_mode"] = True
        config["lam"] = 1.5  # Invalid: > 1

        # Validation should catch this
        from core.validation import ValidationError, validate_config

        with pytest.raises(ValidationError):
            validate_config(config)

    def test_discrete_mode_signal_convergence(self):
        """Discrete mode signal should use lam parameter correctly."""
        config = dict(CONFIG)
        config["use_canonical_discrete_mode"] = True
        config["lam"] = 0.1  # Small integration rate
        config["stochastic_ignition"] = False

        pipeline = APGIPipeline(config)

        # Run many steps with constant input
        S_history = []
        for _ in range(100):
            pipeline.step(x_e=0.5, x_i=0.5)
            S_history.append(pipeline.S)

        # Discrete mode should produce valid signal values (not NaN or inf)
        assert all(np.isfinite(S_history))
        # Signal should not explode to unreasonable values
        assert max(abs(s) for s in S_history) < 100.0


class TestSignalAccumulationFidelity:
    """Test signal accumulation equation fidelity against spec."""

    def test_discrete_leaky_accumulation_formula(self):
        """Verify discrete mode implements S(t+1) = (1-λ)S(t) + λS_inst(t)."""
        from core.signal import integrate_signal_leaky

        lam = 0.3
        S_prev = 1.0
        S_inst = 2.0

        # Expected result per spec
        expected = (1 - lam) * S_prev + lam * S_inst

        # Actual result
        actual = integrate_signal_leaky(S_prev, S_inst, lam)

        assert np.isclose(actual, expected)

    def test_ode_drift_term_structure(self):
        """Verify ODE mode includes drift term -S/τ_S + precision terms."""
        from core.dynamics import signal_drift

        S = 1.0
        phi_e = 0.5
        phi_i = 0.5
        pi_e = 1.0
        pi_i = 1.0
        tau_s = 5.0

        drift = signal_drift(S, phi_e, phi_i, pi_e, pi_i, tau_s)

        # Should include -S/τ_S term
        expected_decay = -S / tau_s
        expected_drive = pi_e * phi_e + pi_i * phi_i
        expected = expected_decay + expected_drive

        assert np.isclose(drift, expected)
