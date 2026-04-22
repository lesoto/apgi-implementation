"""Comprehensive unit tests for core/sde.py module.

Tests cover:
- integrate_euler_maruyama function
"""

from __future__ import annotations

import numpy as np
import pytest

from core.sde import integrate_euler_maruyama


class TestIntegrateEulerMaruyama:
    """Tests for integrate_euler_maruyama function."""

    def test_basic_integration(self):
        """Should integrate correctly with constant drift and diffusion."""
        np.random.seed(42)
        result = integrate_euler_maruyama(
            x=1.0,
            mu=0.1,
            sigma=0.2,
            t=0.0,
            dt=1.0,
        )
        # x_new = x + mu*dt + sigma*sqrt(dt)*N(0,1)
        # Deterministic part: 1.0 + 0.1*1.0 = 1.1
        # Stochastic part depends on random sample
        assert result != 1.0  # Should have changed

    def test_zero_drift(self):
        """Should only add noise when drift is zero."""
        np.random.seed(42)
        result = integrate_euler_maruyama(
            x=1.0,
            mu=0.0,
            sigma=0.1,
            t=0.0,
            dt=1.0,
        )
        # Only stochastic part
        assert result != 1.0

    def test_zero_diffusion(self):
        """Should be deterministic when diffusion is zero."""
        result = integrate_euler_maruyama(
            x=1.0,
            mu=0.1,
            sigma=0.0,
            t=0.0,
            dt=1.0,
        )
        # Deterministic: 1.0 + 0.1*1.0 = 1.1
        assert result == 1.1

    def test_zero_dt(self):
        """Should raise ValueError for non-positive dt."""
        with pytest.raises(ValueError, match="dt must be > 0"):
            integrate_euler_maruyama(1.0, 0.1, 0.1, 0.0, dt=0.0)

        with pytest.raises(ValueError, match="dt must be > 0"):
            integrate_euler_maruyama(1.0, 0.1, 0.1, 0.0, dt=-0.1)

    def test_small_dt(self):
        """Should make smaller changes with smaller dt."""
        np.random.seed(42)
        result_small = integrate_euler_maruyama(
            x=1.0,
            mu=0.1,
            sigma=0.1,
            t=0.0,
            dt=0.01,
        )

        np.random.seed(42)
        result_large = integrate_euler_maruyama(
            x=1.0,
            mu=0.1,
            sigma=0.1,
            t=0.0,
            dt=1.0,
        )

        # Change should be smaller with smaller dt
        change_small = abs(result_small - 1.0)
        change_large = abs(result_large - 1.0)
        assert change_small < change_large

    def test_callable_drift(self):
        """Should handle callable drift function."""

        def drift_fn(x, t):
            return 0.1 * x  # Proportional drift

        np.random.seed(42)
        result = integrate_euler_maruyama(
            x=2.0,
            mu=drift_fn,
            sigma=0.0,
            t=0.0,
            dt=1.0,
        )
        # Drift = 0.1 * 2.0 = 0.2
        # x_new = 2.0 + 0.2*1.0 = 2.2
        assert result == 2.2

    def test_callable_diffusion(self):
        """Should handle callable diffusion function."""

        def diffusion_fn(x, t):
            return 0.1 * abs(x)  # Proportional to |x|

        np.random.seed(42)
        result = integrate_euler_maruyama(
            x=2.0,
            mu=0.0,
            sigma=diffusion_fn,
            t=0.0,
            dt=1.0,
        )
        # Result should be close to 2.0 but with some noise
        assert abs(result - 2.0) < 1.0  # Should be within reasonable range

    def test_reproducibility_with_seed(self):
        """Should be reproducible with same seed."""
        np.random.seed(42)
        result1 = integrate_euler_maruyama(
            x=1.0,
            mu=0.1,
            sigma=0.2,
            t=0.0,
            dt=1.0,
        )

        np.random.seed(42)
        result2 = integrate_euler_maruyama(
            x=1.0,
            mu=0.1,
            sigma=0.2,
            t=0.0,
            dt=1.0,
        )

        assert result1 == result2

    def test_negative_drift(self):
        """Should handle negative drift."""
        result = integrate_euler_maruyama(
            x=1.0,
            mu=-0.1,
            sigma=0.0,
            t=0.0,
            dt=1.0,
        )
        # x_new = 1.0 - 0.1*1.0 = 0.9
        assert result == 0.9
