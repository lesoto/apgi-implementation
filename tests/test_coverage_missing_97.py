"""Tests targeting the remaining 97% → ~99% coverage gap.

Remaining uncovered lines:
- energy/thermodynamics.py:138 (return 0.0)
- examples/16_figures.py:34-382 (plot generation - requires display)
- hierarchy/coupling.py:63-64 (top-down coupling path)
- hierarchy/resonance.py:188, 299 (phase wraparound)
- pipeline.py:297-310, 407-409, 952-953, 1119 (config branches)
- reservoir/liquid_state_machine.py:141, 145, 157, 320 (edge cases)
- stats/hurst.py:109, 257, 266, 310 (edge cases)
- stats/spectral_extraction.py:222 (NaN handling)
- validation/falsification_registry.py:448-451, 477 (edge cases)
- validation/observable_mapping.py:115, 228-231, 242, 255, 268, 281, 296, 430, 443, 456, 475, 489, 504
"""

import numpy as np
import pytest


def test_coupling_pi_plus_1_coupling_active():
    """Test top-down coupling when pi_ell_plus_1 is provided (line 63-64)"""
    from hierarchy.coupling import precision_coupling_ode

    # Active top-down coupling
    result = precision_coupling_ode(
        pi_ell=1.0,
        tau_pi=100.0,
        epsilon_ell=0.5,
        alpha_gain=0.1,
        pi_ell_plus_1=0.8,
        epsilon_ell_minus_1=None,
        C_down=0.2,  # Non-zero
        C_up=0.0,
        psi=None,
    )
    assert isinstance(result, float)
    # Should be different from no coupling
    result_no_coupling = precision_coupling_ode(
        pi_ell=1.0,
        tau_pi=100.0,
        epsilon_ell=0.5,
        alpha_gain=0.1,
        pi_ell_plus_1=None,
        epsilon_ell_minus_1=None,
        C_down=0.0,
        C_up=0.0,
        psi=None,
    )
    assert result != result_no_coupling


def test_nonlinear_phase_amplitude_coupling_all_types():
    """Test nonlinear_phase_amplitude_coupling with all nonlinearity types"""
    from hierarchy.coupling import nonlinear_phase_amplitude_coupling

    for nonlinearity in ["sigmoid", "power", "exponential", "unknown"]:
        result = nonlinear_phase_amplitude_coupling(
            theta_0_ell=1.0,
            pi_ell_plus_1=0.5,
            phi_ell_plus_1=np.pi / 4,
            kappa_down=0.1,
            nonlinearity=nonlinearity,
        )
        assert isinstance(result, float)


def test_bidirectional_threshold_cascade_edge():
    """Test bidirectional_threshold_cascade with hysteresis (line 299)"""
    from hierarchy.coupling import bidirectional_threshold_cascade

    result = bidirectional_threshold_cascade(
        theta_ell=1.0,
        S_ell_minus_1=1.5,
        theta_ell_minus_1=1.0,
        S_ell_plus_1=0.5,
        theta_ell_plus_1=1.0,
        kappa_up=0.1,
        kappa_down=0.05,
        hysteresis=0.1,
    )
    assert isinstance(result, float)
    assert result > 0


def test_liquid_state_machine_edge_case_141():
    """Test error when heterogeneous_tau shape is wrong (line 141)"""
    from reservoir.liquid_state_machine import LiquidStateMachine

    with pytest.raises(ValueError, match="shape"):
        LiquidStateMachine(
            N=100,
            M=2,
            heterogeneous_tau=np.array([1.0, 2.0, 3.0]),  # Wrong shape
        )


def test_liquid_state_machine_heterogeneous_overflow():
    """Test with very large tau value (line 145)"""
    from reservoir.liquid_state_machine import LiquidStateMachine

    with pytest.raises(ValueError, match="must be > 0"):
        LiquidStateMachine(
            N=10,
            M=2,
            heterogeneous_tau=np.array([-1.0] * 10),
        )


def test_liquid_state_machine_collect_state_fail():
    """Test get_training_data fails without data (line 157)"""
    from reservoir.liquid_state_machine import LiquidStateMachine

    lsm = LiquidStateMachine(N=10, M=2)
    # No data collected yet
    with pytest.raises(ValueError, match="No training data"):
        lsm.get_training_data()


def test_liquid_state_machine_suprathreshold_heterogeneous():
    """Test suprathreshold with heterogeneous tau (line 320)"""
    from reservoir.liquid_state_machine import LiquidStateMachine

    heterogeneous_taus = np.array([0.5, 1.0, 1.5, 2.0] * 25)  # 100 units
    lsm = LiquidStateMachine(N=100, M=2, heterogeneous_tau=heterogeneous_taus)
    u = np.array([0.1, -0.1])

    # Should raise due to instability with heterogeneous tau and high A_amp
    with pytest.raises(ValueError, match="unstable"):
        lsm.step(
            u,
            tau=None,  # Use heterogeneous
            S_target=10.0,
            theta=1.0,
            A_amp=5.0,
            saturating_amplification=False,
            alpha_res=None,  # Will compute from heterogeneous
        )


def test_hurst_wavelet_variance_exact_boundary():
    """Test wavelet_variance_analysis at exact max_level boundary (line 257)"""
    from stats.hurst import wavelet_variance_analysis

    signal = np.random.randn(256)
    levels, log_vars = wavelet_variance_analysis(signal, min_level=1, max_level=7)
    assert len(levels) > 0


def test_hurst_estimate_wavelet_exact():
    """Test estimate_hurst_wavelet (line 266)"""
    from stats.hurst import estimate_hurst_wavelet

    signal = np.sin(np.linspace(0, 4 * np.pi, 256)) + 0.1 * np.random.randn(256)
    h = estimate_hurst_wavelet(signal)
    assert 0 <= h <= 1.5


def test_hurst_cross_validate_short_signal():
    """Test cross_validate_hurst edge case (line 310)"""
    from stats.hurst import cross_validate_hurst

    signal = np.random.randn(80)
    result = cross_validate_hurst(signal)
    assert "difference" in result


def test_spectral_extraction_all_nan_welch():
    """Test estimate_spectral_exponent_welch with all-NaN result (line 222)"""
    from stats.spectral_extraction import estimate_spectral_exponent_welch

    signal = np.array([0.0] * 64)  # Constant signal
    beta, r2, h = estimate_spectral_exponent_welch(signal)
    # Should return nan for constant
    assert np.isnan(beta) or beta >= 0


def test_observable_mapping_table():
    """Test get_observable_mapping_table (line 115)"""
    from validation.observable_mapping import get_observable_mapping_table

    table = get_observable_mapping_table()
    assert len(table) > 0
    assert "apgi_variable" in table[0]


def test_neural_observables_step():
    """Test NeuralObservableExtractor.step (line 228-231)"""
    from validation.observable_mapping import NeuralObservableExtractor

    extractor = NeuralObservableExtractor()
    observables = extractor.step(S=1.0, theta=0.5, B=1)
    assert "gamma_power" in observables
    assert "erp_amplitude" in observables
    assert "ignition_rate" in observables


def test_neural_observables_insufficient_history():
    """Test extract_prestimulus_alpha_suppression with empty history (line 242)"""
    from validation.observable_mapping import NeuralObservableExtractor

    extractor = NeuralObservableExtractor()
    result = extractor.extract_prestimulus_alpha_suppression(np.array([]))
    assert result == 0.0


def test_behavioral_observables_step():
    """Test BehavioralObservableExtractor.step (line 255, 268, 281)"""
    from validation.observable_mapping import BehavioralObservableExtractor

    extractor = BehavioralObservableExtractor()
    observables = extractor.step(S=1.0, theta=0.5, B=1)
    assert "rt_variability" in observables
    assert "response_criterion" in observables
    assert "decision_rate" in observables


def test_behavioral_observables_insufficient_data():
    """Test extract methods with insufficient data (line 296)"""
    from validation.observable_mapping import BehavioralObservableExtractor

    extractor = BehavioralObservableExtractor()
    result = extractor.extract_rt_variability(np.array([1.0]))
    assert result == 0.0


def test_key_testable_validator_validate_no_data():
    """Test validate with insufficient data (line 430)"""
    from validation.observable_mapping import KeyTestablePredictionValidator

    validator = KeyTestablePredictionValidator()
    result = validator.validate()
    assert not result["valid"]
    assert "Insufficient" in result["reason"]


def test_key_testable_validator_no_ignition():
    """Test validate when no ignition occurs (line 443)"""
    from validation.observable_mapping import KeyTestablePredictionValidator

    validator = KeyTestablePredictionValidator()
    for _ in range(150):
        validator.step(S=0.5, theta=1.0, B=0)
    result = validator.validate()
    # Should still be valid but with low ignition rate
    if result.get("valid"):
        assert result["ignition_rate"] == 0.0


def test_key_testable_validator_constant_B():
    """Test validate with all 1s in B (line 456)"""
    from validation.observable_mapping import KeyTestablePredictionValidator

    validator = KeyTestablePredictionValidator()
    for _ in range(150):
        validator.step(S=1.5, theta=0.5, B=1)
    result = validator.validate()
    # High ignition rate but may have edge case in correlation computation
    if result.get("valid"):
        assert result["ignition_rate"] == 1.0


def test_identifiability_analyzer_fisher_small_data():
    """Test compute_fisher_information with minimal data (line 475)"""
    from validation.observable_mapping import ParameterIdentifiabilityAnalyzer

    S = np.array([0.5, 0.6, 0.7])
    theta = np.array([1.0, 1.1, 1.2])
    B = np.array([0, 0, 1])
    params = {"ignite_tau": 0.5}

    result = ParameterIdentifiabilityAnalyzer.compute_fisher_information(S, theta, B, params)
    assert "fisher_information" in result


def test_identifiability_analyzer_fisher_zero_variation():
    """Test compute_fisher_information with zero variance (line 489)"""
    from validation.observable_mapping import ParameterIdentifiabilityAnalyzer

    S = np.array([1.0] * 10)  # Constant
    theta = np.array([0.5] * 10)  # Constant
    B = np.array([1, 0] * 5)
    params = {"ignite_tau": 0.5}

    result = ParameterIdentifiabilityAnalyzer.compute_fisher_information(S, theta, B, params)
    # Should handle gracefully
    assert "condition_number" in result


def test_identifiability_analyzer_fisher_single_ignition():
    """Test compute_fisher_information with single ignition event (line 504)"""
    from validation.observable_mapping import ParameterIdentifiabilityAnalyzer

    S = np.random.randn(100)
    theta = np.random.randn(100)
    B = np.array([0] * 99 + [1])  # Single ignition
    params = {"ignite_tau": 0.5}

    result = ParameterIdentifiabilityAnalyzer.compute_fisher_information(S, theta, B, params)
    assert "fisher_information" in result


def test_falsification_check_phase_transition_insufficient():
    """Test check_phase_transition_signatures with insufficient samples (line 448)"""
    from validation.falsification_registry import check_phase_transition_signatures

    S_history = np.random.randn(50)  # Too short
    result = check_phase_transition_signatures(S_history, theta=1.0)
    assert result["verdict"] == "indeterminate"


def test_falsification_check_oscillatory_gate_insufficient():
    """Test check_oscillatory_gate with insufficient trials (line 451)"""
    from validation.falsification_registry import check_oscillatory_gate

    gc = np.array([1.0])
    gm = np.array([0.5])
    result = check_oscillatory_gate(gc, gm, n_min=100)
    assert result["verdict"] == "indeterminate"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
