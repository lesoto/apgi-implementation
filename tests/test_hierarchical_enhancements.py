"""Tests for hierarchical architecture enhancements (Phase 1 - §8).

Tests cover:
- Nonlinear phase-amplitude coupling (PAC)
- Bidirectional phase coupling
- Bidirectional threshold cascade with hysteresis
- Adaptive timescale estimation
- Hierarchy level estimation from data
"""

import numpy as np
import pytest

from hierarchy.coupling import (
    bidirectional_phase_coupling,
    bidirectional_threshold_cascade,
    nonlinear_phase_amplitude_coupling,
)
from hierarchy.multiscale import (
    build_timescales,
    estimate_hierarchy_levels_from_data,
    estimate_optimal_timescale_ratio,
)


class TestNonlinearPAC:
    """Test nonlinear phase-amplitude coupling."""

    def test_nonlinear_pac_sigmoid(self):
        """Test sigmoid nonlinearity in PAC."""
        theta_0 = 1.0
        pi = 0.5
        phi = 0.0  # cos(0) = 1
        kappa = 0.1

        theta_mod = nonlinear_phase_amplitude_coupling(
            theta_0, pi, phi, kappa, nonlinearity="sigmoid"
        )

        # Should be > theta_0 due to positive modulation
        assert theta_mod > theta_0

    def test_nonlinear_pac_power(self):
        """Test power-law nonlinearity in PAC."""
        theta_0 = 1.0
        pi = 0.5
        phi = 0.0
        kappa = 0.1

        theta_mod = nonlinear_phase_amplitude_coupling(
            theta_0, pi, phi, kappa, nonlinearity="power"
        )

        assert theta_mod > 0
        assert isinstance(theta_mod, float)

    def test_nonlinear_pac_exponential(self):
        """Test exponential nonlinearity in PAC."""
        theta_0 = 1.0
        pi = 0.5
        phi = 0.0
        kappa = 0.1

        theta_mod = nonlinear_phase_amplitude_coupling(
            theta_0, pi, phi, kappa, nonlinearity="exponential"
        )

        assert theta_mod > 0
        assert isinstance(theta_mod, float)

    def test_nonlinear_pac_frequency_coupling(self):
        """Test frequency-amplitude coupling strength modulation."""
        theta_0 = 1.0
        pi = 0.5
        phi = np.pi / 4  # sin(π/4) ≈ 0.707
        kappa = 0.1

        # Without frequency coupling
        theta_no_fac = nonlinear_phase_amplitude_coupling(
            theta_0, pi, phi, kappa, phase_frequency_coupling=0.0
        )

        # With frequency coupling
        theta_with_fac = nonlinear_phase_amplitude_coupling(
            theta_0, pi, phi, kappa, phase_frequency_coupling=0.5
        )

        # With FAC should have different modulation
        assert theta_with_fac != theta_no_fac

    def test_nonlinear_pac_phase_dependence(self):
        """Test that PAC depends on phase."""
        theta_0 = 1.0
        pi = 0.5
        kappa = 0.1

        # Phase = 0: cos(0) = 1 (maximum modulation)
        theta_phase_0 = nonlinear_phase_amplitude_coupling(theta_0, pi, 0.0, kappa)

        # Phase = π: cos(π) = -1 (minimum modulation)
        theta_phase_pi = nonlinear_phase_amplitude_coupling(theta_0, pi, np.pi, kappa)

        # Should be different
        assert theta_phase_0 != theta_phase_pi
        assert theta_phase_0 > theta_phase_pi


class TestBidirectionalPhaseCoupling:
    """Test bidirectional Kuramoto phase coupling."""

    def test_bidirectional_phase_coupling_top_down(self):
        """Test top-down phase coupling."""
        phi_ell = 0.0
        phi_ell_plus_1 = np.pi / 2
        omega_ell = 0.1
        dt = 0.1
        kappa_down = 0.5

        phi_new = bidirectional_phase_coupling(
            phi_ell,
            phi_ell_plus_1,
            None,
            omega_ell,
            dt,
            kappa_down=kappa_down,
            kappa_up=0.0,
        )

        # Phase should change due to coupling
        assert phi_new != phi_ell
        assert 0 <= phi_new < 2 * np.pi

    def test_bidirectional_phase_coupling_bottom_up(self):
        """Test bottom-up phase coupling."""
        phi_ell = 0.0
        phi_ell_minus_1 = np.pi / 2
        omega_ell = 0.1
        dt = 0.1
        kappa_up = 0.5

        phi_new = bidirectional_phase_coupling(
            phi_ell,
            None,
            phi_ell_minus_1,
            omega_ell,
            dt,
            kappa_down=0.0,
            kappa_up=kappa_up,
        )

        assert phi_new != phi_ell
        assert 0 <= phi_new < 2 * np.pi

    def test_bidirectional_phase_coupling_both_directions(self):
        """Test simultaneous top-down and bottom-up coupling."""
        phi_ell = 0.0
        phi_ell_plus_1 = np.pi / 2
        phi_ell_minus_1 = np.pi / 4
        omega_ell = 0.1
        dt = 0.1

        phi_new = bidirectional_phase_coupling(
            phi_ell,
            phi_ell_plus_1,
            phi_ell_minus_1,
            omega_ell,
            dt,
            kappa_down=0.3,
            kappa_up=0.2,
        )

        assert 0 <= phi_new < 2 * np.pi

    def test_bidirectional_phase_coupling_with_noise(self):
        """Test phase coupling with stochastic noise."""
        phi_ell = 0.0
        omega_ell = 0.1
        dt = 0.1

        # Multiple runs should give different results due to noise
        results = []
        for _ in range(10):
            phi_new = bidirectional_phase_coupling(
                phi_ell, None, None, omega_ell, dt, noise_std=0.1
            )
            results.append(phi_new)

        # Should have variation
        assert len(set(results)) > 1


class TestBidirectionalThresholdCascade:
    """Test bidirectional threshold cascade with hysteresis."""

    def test_bidirectional_cascade_bottom_up_only(self):
        """Test bottom-up cascade suppression."""
        theta = 1.0
        S_lower = 2.0
        theta_lower = 1.0

        theta_mod = bidirectional_threshold_cascade(
            theta, S_lower, theta_lower, None, None, kappa_up=0.2, kappa_down=0.0
        )

        # Lower level superthreshold should suppress this level
        assert theta_mod < theta

    def test_bidirectional_cascade_top_down_only(self):
        """Test top-down cascade facilitation."""
        theta = 1.0
        S_upper = 2.0
        theta_upper = 1.0

        theta_mod = bidirectional_threshold_cascade(
            theta, None, None, S_upper, theta_upper, kappa_up=0.0, kappa_down=0.2
        )

        # Upper level superthreshold should facilitate this level
        assert theta_mod > theta

    def test_bidirectional_cascade_both_directions(self):
        """Test simultaneous top-down and bottom-up cascade."""
        theta = 1.0
        S_lower = 2.0
        theta_lower = 1.0
        S_upper = 2.0
        theta_upper = 1.0

        theta_mod = bidirectional_threshold_cascade(
            theta,
            S_lower,
            theta_lower,
            S_upper,
            theta_upper,
            kappa_up=0.2,
            kappa_down=0.2,
        )

        # Effects should partially cancel
        assert 0.8 < theta_mod < 1.2

    def test_bidirectional_cascade_hysteresis(self):
        """Test hysteresis prevents oscillations."""
        theta = 1.0
        S_lower = 1.05  # Just above threshold
        theta_lower = 1.0

        # Without hysteresis
        theta_no_hyst = bidirectional_threshold_cascade(
            theta, S_lower, theta_lower, None, None, kappa_up=0.2, hysteresis=0.0
        )

        # With hysteresis
        theta_with_hyst = bidirectional_threshold_cascade(
            theta, S_lower, theta_lower, None, None, kappa_up=0.2, hysteresis=0.1
        )

        # Hysteresis should reduce or prevent suppression
        assert theta_with_hyst >= theta_no_hyst

    def test_bidirectional_cascade_subthreshold(self):
        """Test cascade when signals are subthreshold."""
        theta = 1.0
        S_lower = 0.5
        theta_lower = 1.0

        theta_mod = bidirectional_threshold_cascade(
            theta, S_lower, theta_lower, None, None, kappa_up=0.2
        )

        # Subthreshold should not suppress
        assert theta_mod == theta


class TestAdaptiveTimescaleEstimation:
    """Test adaptive timescale ratio estimation."""

    def test_estimate_timescale_ratio_pink_noise(self):
        """Test timescale ratio estimation on pink noise."""
        # Generate pink noise
        n = 10000
        freqs = np.fft.rfftfreq(n)
        amplitudes = 1.0 / np.sqrt(np.maximum(freqs, 1e-6))
        phases = np.random.uniform(0, 2 * np.pi, len(amplitudes))
        fft = amplitudes * np.exp(1j * phases)
        signal = np.fft.irfft(fft, n)

        k_opt = estimate_optimal_timescale_ratio(signal, fs=1.0, n_levels=4)

        assert 1.3 <= k_opt <= 2.0
        assert isinstance(k_opt, float)

    def test_estimate_timescale_ratio_white_noise(self):
        """Test timescale ratio estimation on white noise."""
        signal = np.random.randn(10000)

        k_opt = estimate_optimal_timescale_ratio(signal, fs=1.0, n_levels=4)

        assert 1.3 <= k_opt <= 2.0

    def test_estimate_timescale_ratio_different_levels(self):
        """Test that ratio estimation works for different level counts."""
        signal = np.random.randn(10000)

        for n_levels in [2, 3, 4, 5]:
            k_opt = estimate_optimal_timescale_ratio(signal, fs=1.0, n_levels=n_levels)
            assert 1.3 <= k_opt <= 2.0


class TestHierarchyLevelEstimation:
    """Test automatic hierarchy level estimation from data."""

    def test_estimate_levels_pink_noise(self):
        """Test level estimation on pink noise."""
        # Generate pink noise
        n = 10000
        freqs = np.fft.rfftfreq(n)
        amplitudes = 1.0 / np.sqrt(np.maximum(freqs, 1e-6))
        phases = np.random.uniform(0, 2 * np.pi, len(amplitudes))
        fft = amplitudes * np.exp(1j * phases)
        signal = np.fft.irfft(fft, n)

        n_levels = estimate_hierarchy_levels_from_data(signal, fs=1.0)

        assert 2 <= n_levels <= 8
        assert isinstance(n_levels, int)

    def test_estimate_levels_white_noise(self):
        """Test level estimation on white noise."""
        signal = np.random.randn(10000)

        n_levels = estimate_hierarchy_levels_from_data(signal, fs=1.0)

        assert 2 <= n_levels <= 8

    def test_estimate_levels_short_signal(self):
        """Test level estimation on short signal."""
        signal = np.random.randn(1000)

        n_levels = estimate_hierarchy_levels_from_data(signal, fs=1.0)

        assert 2 <= n_levels <= 8


class TestIntegration:
    """Integration tests combining multiple enhancements."""

    def test_adaptive_hierarchy_setup(self):
        """Test setting up adaptive hierarchy from data."""
        # Generate signal
        signal = np.random.randn(10000)

        # Estimate parameters
        k_opt = estimate_optimal_timescale_ratio(signal, fs=1.0, n_levels=4)
        n_levels = estimate_hierarchy_levels_from_data(signal, fs=1.0)

        # Build timescales
        taus = build_timescales(tau0=0.01, k=k_opt, n_levels=n_levels)

        assert len(taus) == n_levels
        assert np.all(taus > 0)
        assert np.all(np.diff(taus) > 0)  # Monotonically increasing

    def test_nonlinear_pac_with_bidirectional_coupling(self):
        """Test nonlinear PAC combined with bidirectional coupling."""
        # Simulate hierarchical system
        n_steps = 100
        n_levels = 3

        phi = np.zeros(n_levels)
        theta = np.ones(n_levels)

        for t in range(n_steps):
            # Update phases with bidirectional coupling
            for ell in range(n_levels):
                phi_up = phi[ell + 1] if ell < n_levels - 1 else None
                phi_down = phi[ell - 1] if ell > 0 else None

                phi[ell] = bidirectional_phase_coupling(
                    phi[ell], phi_up, phi_down, 0.1, 0.01, kappa_down=0.1, kappa_up=0.05
                )

            # Update thresholds with nonlinear PAC
            for ell in range(n_levels):
                if ell < n_levels - 1:
                    theta[ell] = nonlinear_phase_amplitude_coupling(
                        1.0, 0.5, phi[ell + 1], 0.1, nonlinearity="sigmoid"
                    )

        # Should complete without errors
        assert len(phi) == n_levels
        assert len(theta) == n_levels


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
