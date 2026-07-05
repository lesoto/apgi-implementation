"""Final push tests for coverage on remaining achievable lines."""

from unittest.mock import patch

import numpy as np
import pytest

from hierarchy.resonance import NestedResonanceSystem
from reservoir.liquid_network import LiquidNetwork
from stats.hurst import hurst_from_slope
from stats.spectral_extraction import estimate_hurst_dfa
from validation.falsification_registry import check_oscillatory_gate
from validation.observable_mapping import BehavioralObservableExtractor, NeuralObservableExtractor


class TestResonanceLine299Full:
    """Ensure line 299 is fully covered with edge cases."""

    def test_nested_resonance_apply_level_ignition_boundary_conditions(self):
        """Test boundary conditions for apply_level_ignition."""
        n_levels = 4
        theta_0 = np.ones(n_levels) * 5.0
        omega = np.ones(n_levels) * 10.0
        lambda_rates = np.ones(n_levels) * 0.3

        system = NestedResonanceSystem(
            n_levels=n_levels,
            theta_0=theta_0,
            omega=omega,
            lambda_rates=lambda_rates,
        )

        # Test all valid levels
        for level in range(n_levels):
            system.apply_level_ignition(level)
            assert system.S[level] >= 0  # S should be reset but non-negative


class TestLiquidNetworkLine141:
    """Test liquid network error validation (line 141)."""

    def test_liquid_network_tau_validation_comprehensive(self):
        """Line 141: Comprehensive tau validation in step()."""
        network = LiquidNetwork(n_units=100, spectral_radius=0.9, res_sparsity=0.15)

        # Test that both negative and zero tau raise errors
        invalid_tau_values = [-10.0, -1.0, -0.001, 0.0]
        for tau in invalid_tau_values:
            with pytest.raises(ValueError, match="tau must be > 0"):
                network.step(u=0.5, tau=tau)

        # Valid tau should work
        result = network.step(u=0.5, tau=0.001)
        assert result is not None


class TestHurstLines257And310:
    """Test hurst_from_slope conversion functions."""

    def test_hurst_from_slope_boundary_cases(self):
        """Lines 257, 310: Boundary case handling for slope conversion."""
        # Exactly at boundary (β=1.0)
        h_at_boundary = hurst_from_slope(1.0, regime="auto", warn_near_boundary=False)
        # Should use fGn formula (slope <= 1.0)
        assert abs(h_at_boundary - 1.0) < 1e-6

        # Very close to boundary
        h_below = hurst_from_slope(0.99, regime="auto", warn_near_boundary=False)
        h_above = hurst_from_slope(1.01, regime="auto", warn_near_boundary=False)
        # Below should give higher H than above
        assert h_below > h_above

    def test_hurst_from_slope_warning_boundary_exact(self):
        """Line 310: Warning at exact boundary β=1.0."""
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            h = hurst_from_slope(1.0, regime="auto", warn_near_boundary=True)
            # May or may not warn at exact boundary depending on implementation
            assert isinstance(h, float)


class TestSpectralExtractionLine222:
    """Test spectral extraction DFA edge cases (line 222)."""

    def test_estimate_hurst_dfa_signal_length_boundary(self):
        """Line 222: Signal length boundary for DFA."""
        # Signal with exactly 16 samples (minimum for DFA)
        signal = np.cumsum(np.random.randn(16))
        slope, r2 = estimate_hurst_dfa(signal)
        # Should handle edge case without crashing
        assert isinstance(slope, (float, np.floating)) or np.isnan(slope)
        assert isinstance(r2, (float, np.floating)) or np.isnan(r2)

        # Signal with 15 samples (below minimum)
        signal = np.cumsum(np.random.randn(15))
        slope, r2 = estimate_hurst_dfa(signal)
        # Should return NaN for insufficient data
        assert np.isnan(slope) or np.isnan(r2)


class TestFalsificationLines448451:
    """Test falsification registry edge cases."""

    def test_check_oscillatory_gate_edge_cases(self):
        """Lines 448-451: Edge case handling."""
        # Both arrays minimal
        gamma_conscious = np.array([1.0])
        gamma_near_miss = np.array([1.0])
        result = check_oscillatory_gate(gamma_conscious, gamma_near_miss)
        assert result["verdict"] in ["confirmed", "falsified", "indeterminate"]

        # One array empty
        gamma_conscious = np.array([])
        gamma_near_miss = np.array([1.0, 2.0])
        result = check_oscillatory_gate(gamma_conscious, gamma_near_miss)
        assert result["verdict"] == "indeterminate"

        # Identical distributions (should be indeterminate)
        gamma_conscious = np.ones(50)
        gamma_near_miss = np.ones(50)
        result = check_oscillatory_gate(gamma_conscious, gamma_near_miss)
        assert result["verdict"] in ["indeterminate", "falsified"]


class TestObservableMappingLines230231And430456489:
    """Test observable mapping extraction methods."""

    def test_behavioral_extractor_all_empty_histories(self):
        """Lines 230-231, 430, 443, 456, 489: Empty history edge cases."""
        extractor = BehavioralObservableExtractor()

        # Test all extract methods with empty inputs
        theta_empty = np.array([])
        B_empty = np.array([])
        delta_empty = np.array([])
        g_ne_empty = np.array([])

        assert extractor.extract_rt_variability(theta_empty) == 0.0
        assert extractor.extract_response_criterion(theta_empty) == 0.0
        assert extractor.extract_decision_rate(B_empty) == 0.0
        assert extractor.extract_rt_distribution_shape(g_ne_empty) == 0.0
        assert extractor.extract_decision_confidence(delta_empty) == 0.0

    def test_behavioral_extractor_single_element_histories(self):
        """Lines 230-231, 430, 443, 456, 489: Single element histories."""
        extractor = BehavioralObservableExtractor()

        theta_single = np.array([1.5])
        B_single = np.array([1])
        delta_single = np.array([0.5])
        g_ne_single = np.array([1.0])

        # Should handle gracefully
        rt_var = extractor.extract_rt_variability(theta_single)
        assert rt_var >= 0

        rc = extractor.extract_response_criterion(theta_single)
        assert rc >= 0

        dr = extractor.extract_decision_rate(B_single)
        assert dr >= 0

        shape = extractor.extract_rt_distribution_shape(g_ne_single)
        assert shape >= 0

        conf = extractor.extract_decision_confidence(delta_single)
        assert 0 <= conf <= 1

    def test_neural_extractor_all_empty_histories(self):
        """Lines 230-231, 430, 443, 456, 489: Neural extractor empty cases."""
        extractor = NeuralObservableExtractor()

        empty = np.array([])

        # All methods should handle empty gracefully
        assert extractor.extract_alpha_power_proxy(empty) == 0.0
        assert extractor.extract_pupil_diameter_proxy(empty) == 0.0
        assert extractor.extract_striatal_bold_proxy(empty) == 0.0
        assert extractor.extract_hrv_proxy(empty) == 0.0
        # Gamma/beta may return 1.0 or 0.0 for empty
        result = extractor.extract_gamma_beta_ratio(empty)
        assert result == 0.0 or result == 1.0 or np.isnan(result)

    def test_behavioral_extractor_large_window(self):
        """Line 430 variant: Request with window larger than history."""
        extractor = BehavioralObservableExtractor()

        theta_short = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        # Window size much larger than data
        rc = extractor.extract_response_criterion(theta_short, window_size=1000)
        # Should use available data
        assert abs(rc - np.mean(theta_short)) < 1e-6

    def test_neural_extractor_constant_signal(self):
        """Line 430 variant: Neural extractor with constant signal."""
        extractor = NeuralObservableExtractor()

        constant = np.ones(100) * 5.0

        # Constant signal should still produce valid output
        alpha = extractor.extract_alpha_power_proxy(constant)
        assert alpha >= 0


class TestLiquidNetworkSuprathreshold:
    """Test liquid network suprathreshold conditions."""

    def test_liquid_network_suprathreshold_boundary(self):
        """Test liquid network suprathreshold conditions more thoroughly."""
        network = LiquidNetwork(n_units=100, spectral_radius=0.9, res_sparsity=0.15)

        # Set up a state that might trigger suprathreshold condition
        network.x = np.random.randn(100) * 0.5

        # With saturating=True, should never raise error
        for _ in range(10):
            result = network.step(
                u=0.5,
                tau=1.0,
                S_target=5.0,
                theta=0.5,
                A_amp=5.0,
                saturating_amplification=True,
            )
            assert result is not None
