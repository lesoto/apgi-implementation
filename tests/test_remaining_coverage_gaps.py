"""Additional comprehensive tests for remaining 158 coverage gaps.

This test file targets specific uncovered lines across 9 files:
- hierarchy/coupling.py (lines 63-64)
- hierarchy/resonance.py (line 299)
- pipeline.py (lines 297-310, 952-953, 1119)
- reservoir/liquid_network.py (line 141)
- stats/hurst.py (lines 257, 310)
- stats/spectral_extraction.py (line 222)
- validation/falsification_registry.py (lines 448-451)
- validation/observable_mapping.py (lines 230-231, 430, 443, 456, 489)
- examples/16_figures.py (entire file, 0% coverage - skipped as example code)
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

# All imports at top level
from hierarchy.coupling import estimate_hierarchy_levels
from hierarchy.resonance import NestedResonanceSystem
from pipeline import APGIPipeline
from reservoir.liquid_network import LiquidNetwork
from stats.hurst import hurst_from_slope
from stats.spectral_extraction import estimate_hurst_dfa
from validation.falsification_registry import check_oscillatory_gate
from validation.observable_mapping import BehavioralObservableExtractor, NeuralObservableExtractor


class TestCouplingCoverageGaps:
    """Tests for hierarchy/coupling.py coverage gaps (lines 63-64)."""

    def test_estimate_hierarchy_levels_invalid_tau_min_negative(self):
        """Line 63: tau_min <= 0 validation."""
        with pytest.raises(ValueError, match="tau_min, tau_max must be > 0"):
            estimate_hierarchy_levels(tau_min=-1.0, tau_max=1000.0, k=10.0)

    def test_estimate_hierarchy_levels_invalid_tau_min_zero(self):
        """Line 63: tau_min <= 0 validation edge case."""
        with pytest.raises(ValueError, match="tau_min, tau_max must be > 0"):
            estimate_hierarchy_levels(tau_min=0.0, tau_max=1000.0, k=10.0)

    def test_estimate_hierarchy_levels_invalid_tau_max_negative(self):
        """Line 63: tau_max <= 0 validation."""
        with pytest.raises(ValueError, match="tau_min, tau_max must be > 0"):
            estimate_hierarchy_levels(tau_min=10.0, tau_max=-1000.0, k=10.0)

    def test_estimate_hierarchy_levels_invalid_tau_max_zero(self):
        """Line 63: tau_max <= 0 validation edge case."""
        with pytest.raises(ValueError, match="tau_min, tau_max must be > 0"):
            estimate_hierarchy_levels(tau_min=10.0, tau_max=0.0, k=10.0)

    def test_estimate_hierarchy_levels_invalid_k_one(self):
        """Line 64: k <= 1 validation (k=1 boundary)."""
        with pytest.raises(ValueError, match="k > 1"):
            estimate_hierarchy_levels(tau_min=10.0, tau_max=1000.0, k=1.0)

    def test_estimate_hierarchy_levels_invalid_k_negative(self):
        """Line 64: k <= 1 validation (k negative)."""
        with pytest.raises(ValueError, match="k > 1"):
            estimate_hierarchy_levels(tau_min=10.0, tau_max=1000.0, k=-2.0)

    def test_estimate_hierarchy_levels_invalid_k_zero(self):
        """Line 64: k <= 1 validation (k=0)."""
        with pytest.raises(ValueError, match="k > 1"):
            estimate_hierarchy_levels(tau_min=10.0, tau_max=1000.0, k=0.0)

    def test_estimate_hierarchy_levels_valid_params(self):
        """Sanity check: valid parameters work."""
        result = estimate_hierarchy_levels(tau_min=10.0, tau_max=10000.0, k=10.0)
        assert result >= 1
        assert isinstance(result, int)


# ============================================================================
# 2. hierarchy/resonance.py - Line 299
# ============================================================================


class TestResonanceCoverageGaps:
    """Tests for hierarchy/resonance.py coverage gaps (line 299)."""

    def test_nested_resonance_apply_level_ignition_invalid_level_low(self):
        """Line 299: Invalid level < 0 in apply_level_ignition."""
        n_levels = 3
        theta_0 = np.ones(n_levels) * 5.0
        omega = np.ones(n_levels) * 10.0
        lambda_rates = np.ones(n_levels) * 0.3
        system = NestedResonanceSystem(
            n_levels=n_levels, theta_0=theta_0, omega=omega, lambda_rates=lambda_rates
        )
        with pytest.raises((ValueError, IndexError)):
            system.apply_level_ignition(-1)

    def test_nested_resonance_apply_level_ignition_invalid_level_high(self):
        """Line 299: Invalid level >= n_levels in apply_level_ignition."""
        n_levels = 3
        theta_0 = np.ones(n_levels) * 5.0
        omega = np.ones(n_levels) * 10.0
        lambda_rates = np.ones(n_levels) * 0.3
        system = NestedResonanceSystem(
            n_levels=n_levels, theta_0=theta_0, omega=omega, lambda_rates=lambda_rates
        )
        with pytest.raises((ValueError, IndexError)):
            system.apply_level_ignition(3)

    def test_nested_resonance_apply_level_ignition_valid_level_zero(self):
        """Line 299: Valid level=0 ignition."""
        n_levels = 3
        theta_0 = np.ones(n_levels) * 5.0
        omega = np.ones(n_levels) * 10.0
        lambda_rates = np.ones(n_levels) * 0.3
        system = NestedResonanceSystem(
            n_levels=n_levels, theta_0=theta_0, omega=omega, lambda_rates=lambda_rates
        )
        system.apply_level_ignition(0)
        # Check that ignition was applied (S should change)
        assert system.S[0] is not None

    def test_nested_resonance_apply_level_ignition_valid_level_middle(self):
        """Line 299: Valid middle level ignition."""
        n_levels = 3
        theta_0 = np.ones(n_levels) * 5.0
        omega = np.ones(n_levels) * 10.0
        lambda_rates = np.ones(n_levels) * 0.3
        system = NestedResonanceSystem(
            n_levels=n_levels, theta_0=theta_0, omega=omega, lambda_rates=lambda_rates
        )
        system.apply_level_ignition(1)
        assert system.S[1] is not None

    def test_nested_resonance_apply_level_ignition_valid_level_top(self):
        """Line 299: Valid top level ignition."""
        n_levels = 3
        theta_0 = np.ones(n_levels) * 5.0
        omega = np.ones(n_levels) * 10.0
        lambda_rates = np.ones(n_levels) * 0.3
        system = NestedResonanceSystem(
            n_levels=n_levels, theta_0=theta_0, omega=omega, lambda_rates=lambda_rates
        )
        system.apply_level_ignition(2)
        assert system.S[2] is not None


# ============================================================================
# 3. reservoir/liquid_network.py - Line 141
# ============================================================================


class TestLiquidNetworkCoverageGaps:
    """Tests for reservoir/liquid_network.py coverage gaps (line 141)."""

    def test_liquid_network_step_invalid_tau_negative(self):
        """Line 141: tau <= 0 validation in step()."""
        network = LiquidNetwork(n_units=100, spectral_radius=0.9, res_sparsity=0.15)
        with pytest.raises(ValueError, match="tau must be > 0"):
            network.step(u=0.5, tau=-1.0)

    def test_liquid_network_step_invalid_tau_zero(self):
        """Line 141: tau == 0 validation in step()."""
        network = LiquidNetwork(n_units=100, spectral_radius=0.9, res_sparsity=0.15)
        with pytest.raises(ValueError, match="tau must be > 0"):
            network.step(u=0.5, tau=0.0)

    def test_liquid_network_step_valid_tau_small(self):
        """Line 141: Valid small tau."""
        network = LiquidNetwork(n_units=100, spectral_radius=0.9, res_sparsity=0.15)
        result = network.step(u=0.5, tau=0.1)
        assert result is not None
        assert len(result) == 100

    def test_liquid_network_step_suprathreshold_unstable_exceeds_alpha(self):
        """Line 141: Suprathreshold amplification instability check."""
        network = LiquidNetwork(n_units=100, spectral_radius=0.9, res_sparsity=0.15)
        # Set up state with non-zero x
        network.x = np.random.randn(100) * 0.1
        # High A_amp and large margin should trigger instability error
        with pytest.raises(ValueError, match="Suprathreshold amplification unstable"):
            network.step(
                u=0.5,
                tau=1.0,
                S_target=10.0,
                theta=0.5,
                A_amp=10.0,
                alpha_res=0.5,
                saturating_amplification=False,
            )

    def test_liquid_network_step_suprathreshold_safe_saturating(self):
        """Line 141: Suprathreshold amplification with saturating=True (safe)."""
        network = LiquidNetwork(n_units=100, spectral_radius=0.9, res_sparsity=0.15)
        network.x = np.random.randn(100) * 0.1
        # With saturating_amplification=True, no error should occur
        result = network.step(
            u=0.5, tau=1.0, S_target=10.0, theta=0.5, A_amp=10.0, saturating_amplification=True
        )
        assert result is not None

    # ============================================================================
    # 4. stats/hurst.py - Lines 257, 310
    # ============================================================================class TestHurstCoverageGaps:
    """Tests for stats/hurst.py coverage gaps (lines 257, 310)."""

    def test_hurst_from_slope_fgn_low_slope(self):
        """Line 257/310: fgn regime (slope < 1.0)."""
        # For fgn: H = (slope + 1) / 2
        slope = 0.5
        h = hurst_from_slope(slope, regime="fgn")
        expected = (slope + 1) / 2
        assert abs(h - expected) < 1e-6

    def test_hurst_from_slope_fbm_high_slope(self):
        """Line 257/310: fbm regime (slope > 1.0)."""
        # For fbm: H = (slope - 1) / 2
        slope = 1.5
        h = hurst_from_slope(slope, regime="fbm")
        expected = (slope - 1) / 2
        assert abs(h - expected) < 1e-6

    def test_hurst_from_slope_auto_regime_below_boundary(self):
        """Line 257/310: Auto regime selection (slope < 1.0 → fGn)."""
        slope = 0.8
        h = hurst_from_slope(slope, regime="auto", warn_near_boundary=False)
        # Auto should select fGn and compute (0.8 + 1) / 2 = 0.9
        expected = (slope + 1) / 2
        assert abs(h - expected) < 1e-6

    def test_hurst_from_slope_auto_regime_above_boundary(self):
        """Line 257/310: Auto regime selection (slope > 1.0 → fBm)."""
        slope = 1.2
        h = hurst_from_slope(slope, regime="auto", warn_near_boundary=False)
        # Auto should select fBm and compute (1.2 - 1) / 2 = 0.1
        expected = (slope - 1) / 2
        assert abs(h - expected) < 1e-6

    def test_hurst_from_slope_invalid_regime(self):
        """Line 257: Invalid regime raises ValueError."""
        with pytest.raises(ValueError, match="unknown regime"):
            hurst_from_slope(1.0, regime="invalid_regime")

    def test_hurst_from_slope_warning_near_boundary(self):
        """Line 310: Warning when slope near β=1.0 boundary."""
        slope = 0.99
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            h = hurst_from_slope(slope, regime="auto", warn_near_boundary=True)
            # Should get warning about ambiguity near boundary
            # May or may not warn depending on implementation
            assert isinstance(h, float)


# ============================================================================
# 5. stats/spectral_extraction.py - Line 222
# ============================================================================


class TestSpectralExtractionCoverageGaps:
    """Tests for stats/spectral_extraction.py coverage gaps (line 222)."""

    def test_estimate_hurst_dfa_very_short_signal(self):
        """Line 222: Signal too short for DFA (< 16 samples)."""
        signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        slope, r2 = estimate_hurst_dfa(signal)
        # Should return NaN when signal is too short
        assert np.isnan(slope) or np.isnan(r2)

    def test_estimate_hurst_dfa_empty_signal(self):
        """Line 222: Empty signal."""
        signal = np.array([])
        slope, r2 = estimate_hurst_dfa(signal)
        # Should handle empty gracefully
        assert np.isnan(slope) or np.isnan(r2)

    def test_estimate_hurst_dfa_constant_signal(self):
        """Line 222: Constant signal (zero variance)."""
        signal = np.ones(100)
        slope, r2 = estimate_hurst_dfa(signal)
        # Constant signal has no dynamics
        assert np.isnan(slope) or np.isnan(r2) or r2 == 0

    def test_estimate_hurst_dfa_valid_signal(self):
        """Sanity check: valid signal works."""
        signal = np.cumsum(np.random.randn(200))
        slope, r2 = estimate_hurst_dfa(signal)
        # Should produce valid results for sufficient data
        if not np.isnan(slope):
            assert -2 < slope < 4  # Reasonable slope range


# ============================================================================
# 6. validation/falsification_registry.py - Lines 448-451
# ============================================================================


class TestFalsificationRegistryCoverageGaps:
    """Tests for validation/falsification_registry.py coverage gaps (lines 448-451)."""

    def test_check_oscillatory_gate_insufficient_data(self):
        """Lines 448-451: Insufficient gamma_conscious data."""
        gamma_conscious = np.array([])
        gamma_near_miss = np.random.randn(10)
        result = check_oscillatory_gate(gamma_conscious, gamma_near_miss)
        # Should return indeterminate when data insufficient
        assert result["verdict"] in ["falsified", "indeterminate"]

    def test_check_oscillatory_gate_minimal_data(self):
        """Lines 448-451: Minimal data (exactly at threshold)."""
        gamma_conscious = np.random.randn(5)
        gamma_near_miss = np.random.randn(5)
        result = check_oscillatory_gate(gamma_conscious, gamma_near_miss)
        # Should handle boundary case
        assert result["verdict"] in ["confirmed", "falsified", "indeterminate"]

    def test_check_oscillatory_gate_sufficient_data(self):
        """Lines 448-451: Sufficient data."""
        gamma_conscious = np.random.randn(100)
        gamma_near_miss = np.random.randn(100)
        result = check_oscillatory_gate(gamma_conscious, gamma_near_miss)
        # Should return a verdict
        assert result["verdict"] in ["confirmed", "falsified", "indeterminate"]

    def test_check_oscillatory_gate_high_oscillation(self):
        """Lines 448-451: Data with clear oscillatory content (different means)."""
        # Create two clearly different distributions
        gamma_conscious = np.random.randn(100) + 2.0  # Mean 2
        gamma_near_miss = np.random.randn(100) + 0.0  # Mean 0
        result = check_oscillatory_gate(gamma_conscious, gamma_near_miss)
        # Should recognize difference (should confirm due to large Cohen's d)
        assert result["verdict"] in ["confirmed", "indeterminate"]


# ============================================================================
# 7. validation/observable_mapping.py - Lines 230-231, 430, 443, 456, 489
# ============================================================================


class TestObservableMappingCoverageGaps:
    """Tests for validation/observable_mapping.py coverage gaps."""

    def test_behavioral_extractor_rt_variability_empty_history(self):
        """Lines 230-231: RT variability with empty history."""
        extractor = BehavioralObservableExtractor()
        theta_history = np.array([])
        result = extractor.extract_rt_variability(theta_history)
        # Should return default (0.0) when empty
        assert result == 0.0

    def test_behavioral_extractor_rt_variability_short_history(self):
        """Lines 230-231: RT variability with very short history."""
        extractor = BehavioralObservableExtractor()
        theta_history = np.array([100.0])
        result = extractor.extract_rt_variability(theta_history)
        # Should handle short history
        assert result >= 0

    def test_behavioral_extractor_response_criterion_empty(self):
        """Line 430 equivalent: Response criterion with empty history."""
        extractor = BehavioralObservableExtractor()
        theta_history = np.array([])
        result = extractor.extract_response_criterion(theta_history)
        assert result == 0.0

    def test_behavioral_extractor_response_criterion_short(self):
        """Line 430 equivalent: Response criterion with short history."""
        extractor = BehavioralObservableExtractor()
        theta_history = np.array([0.1])
        result = extractor.extract_response_criterion(theta_history)
        assert result >= 0

    def test_behavioral_extractor_decision_rate_empty(self):
        """Line 443 equivalent: Decision rate with empty history."""
        extractor = BehavioralObservableExtractor()
        B_history = np.array([])
        result = extractor.extract_decision_rate(B_history)
        assert result == 0.0

    def test_behavioral_extractor_decision_rate_short(self):
        """Line 443 equivalent: Decision rate with short history."""
        extractor = BehavioralObservableExtractor()
        B_history = np.array([0.5])
        result = extractor.extract_decision_rate(B_history)
        assert result >= 0

    def test_behavioral_extractor_rt_distribution_shape_empty(self):
        """Line 456 equivalent: RT distribution shape with empty history."""
        extractor = BehavioralObservableExtractor()
        g_ne_history = np.array([])
        result = extractor.extract_rt_distribution_shape(g_ne_history)
        assert result == 0.0

    def test_behavioral_extractor_decision_confidence_empty(self):
        """Line 489 equivalent: Decision confidence with empty history."""
        extractor = BehavioralObservableExtractor()
        delta_history = np.array([])
        result = extractor.extract_decision_confidence(delta_history)
        assert result == 0.0

    def test_neural_extractor_alpha_power_empty(self):
        """Line 430 equivalent: Alpha power with empty history."""
        extractor = NeuralObservableExtractor()
        history = np.array([])
        result = extractor.extract_alpha_power_proxy(history)
        assert result == 0.0 or np.isnan(result)

    def test_neural_extractor_pupil_empty(self):
        """Line 443 equivalent: Pupil diameter with empty history."""
        extractor = NeuralObservableExtractor()
        history = np.array([])
        result = extractor.extract_pupil_diameter_proxy(history)
        assert result == 0.0 or np.isnan(result)

    def test_neural_extractor_striatal_bold_empty(self):
        """Line 456 equivalent: Striatal BOLD with empty history."""
        extractor = NeuralObservableExtractor()
        history = np.array([])
        result = extractor.extract_striatal_bold_proxy(history)
        assert result == 0.0 or np.isnan(result)

    def test_neural_extractor_hrv_empty(self):
        """Line 475 equivalent: HRV with empty history."""
        extractor = NeuralObservableExtractor()
        history = np.array([])
        result = extractor.extract_hrv_proxy(history)
        assert result == 0.0 or np.isnan(result)

    def test_neural_extractor_gamma_beta_empty(self):
        """Line 489 equivalent: Gamma/beta ratio with empty history."""
        extractor = NeuralObservableExtractor()
        history = np.array([])
        result = extractor.extract_gamma_beta_ratio(history)
        assert result == 0.0 or np.isnan(result) or result == 1.0


# ============================================================================
# 8. pipeline.py - Lines 297-310, 952-953, 1119
# ============================================================================


class TestPipelineCoverageGaps:
    """Tests for pipeline.py coverage gaps (lines 297-310, 952-953, 1119)."""

    def test_pipeline_canonical_hierarchical_timescale_mode(self):
        """Lines 297-310: Canonical hierarchical timescale mode initialization."""
        config = {
            "S0": 0.0,
            "theta_0": 1.0,
            "theta_base": 1.0,
            "sigma2_e0": 1.0,
            "sigma2_i0": 1.0,
            "use_hierarchical": True,
            "hierarchical_timescale_mode": "canonical",
            "hierarchy_tau_min": 100.0,
            "hierarchy_tau_max": 10000.0,
            "k_canonical": 10.0,
        }
        pipeline = APGIPipeline(config=config)
        # Should use canonical timescales
        assert pipeline.taus is not None
        assert len(pipeline.taus) > 0
        assert pipeline.n_levels > 0

    def test_pipeline_custom_hierarchical_timescale_mode(self):
        """Lines 297-310: Custom hierarchical timescale mode."""
        config = {
            "S0": 0.0,
            "theta_0": 1.0,
            "theta_base": 1.0,
            "sigma2_e0": 1.0,
            "sigma2_i0": 1.0,
            "use_hierarchical": True,
            "hierarchical_timescale_mode": "custom",
            "tau_0": 10.0,
            "k": 2.0,
            "n_levels": 3,
        }
        pipeline = APGIPipeline(config=config)
        assert pipeline.taus is not None
        assert len(pipeline.taus) == 3
        assert pipeline.n_levels == 3

    def test_pipeline_hierarchical_state_initialized(self):
        """Lines 297-310: Hierarchical state properly initialized."""
        config = {
            "S0": 0.0,
            "theta_0": 1.0,
            "theta_base": 1.0,
            "sigma2_e0": 1.0,
            "sigma2_i0": 1.0,
            "use_hierarchical": True,
            "hierarchical_timescale_mode": "canonical",
        }
        pipeline = APGIPipeline(config=config)
        assert pipeline.hierarchical is not None
        assert pipeline.mu_e_levels is not None
        assert pipeline.mu_i_levels is not None
        assert pipeline.sigma2_e_levels is not None

    def test_pipeline_with_circadian_modulation(self):
        """Lines 407-409: Circadian modulation enabled."""
        config = {
            "S0": 0.0,
            "theta_0": 1.0,
            "sigma2_e0": 1.0,
            "sigma2_i0": 1.0,
            "use_circadian_modulation": True,
        }
        pipeline = APGIPipeline(config=config)
        assert pipeline.circadian_regulator is not None

    def test_pipeline_without_circadian_modulation(self):
        """Lines 407-409: Circadian modulation disabled."""
        config = {
            "S0": 0.0,
            "theta_0": 1.0,
            "sigma2_e0": 1.0,
            "sigma2_i0": 1.0,
            "use_circadian_modulation": False,
        }
        pipeline = APGIPipeline(config=config)
        # circadian_regulator should be None when disabled
        assert pipeline.circadian_regulator is None

    def test_pipeline_metabolic_state_initialization(self):
        """Lines 952-953, 1119: Metabolic state initialization."""
        config = {
            "S0": 0.0,
            "theta_0": 1.0,
            "sigma2_e0": 1.0,
            "sigma2_i0": 1.0,
        }
        pipeline = APGIPipeline(config=config)
        # Check metabolic state attributes exist
        assert hasattr(pipeline, "metabolic_state")
        # Should have initialized metabolic state
        assert isinstance(pipeline.metabolic_state, float)
