"""Tests for observable mapping and validation.

Tests spec §14: Observable Mapping
"""

import numpy as np
import pytest

from validation.observable_mapping import (
    BehavioralObservableExtractor,
    KeyTestablePredictionValidator,
    NeuralObservableExtractor,
    ParameterIdentifiabilityAnalyzer,
)


class TestNeuralObservableExtractor:
    """Test neural observable extraction."""

    def test_initialization(self):
        """Test neural extractor initialization."""
        extractor = NeuralObservableExtractor(fs=100.0)
        assert extractor.fs == 100.0
        assert len(extractor.history) == 6

    def test_gamma_power_extraction(self):
        """Test gamma-band power extraction."""
        extractor = NeuralObservableExtractor(fs=100.0)

        # Create synthetic signal with gamma component
        t = np.arange(0, 1, 0.01)  # 1 second at 100 Hz
        signal = np.sin(2 * np.pi * 50 * t)  # 50 Hz (gamma band)

        gamma_power = extractor.extract_gamma_power(signal)

        # Should be positive
        assert gamma_power >= 0

    def test_erp_amplitude_extraction(self):
        """Test ERP amplitude extraction."""
        extractor = NeuralObservableExtractor()

        # Create synthetic threshold history
        theta_history = np.ones(100) * 1.0
        theta_history[50:60] = 2.0  # Peak

        erp_amplitude = extractor.extract_erp_amplitude(theta_history)

        # Should be positive (peak above baseline)
        assert erp_amplitude > 0

    def test_ignition_rate_extraction(self):
        """Test ignition rate extraction."""
        extractor = NeuralObservableExtractor()

        # Create synthetic ignition history
        B_history = np.zeros(100)
        B_history[::10] = 1  # 10% ignition rate

        ignition_rate = extractor.extract_ignition_rate(B_history)

        # Should be close to 0.1
        assert 0.05 < ignition_rate < 0.15

    def test_step(self):
        """Test step update."""
        extractor = NeuralObservableExtractor()

        # Take multiple steps
        for t in range(100):
            S = np.sin(0.1 * t)
            theta = 1.0 + 0.1 * np.cos(0.05 * t)
            B = 1 if S > theta else 0

            result = extractor.step(S, theta, B)

            assert "gamma_power" in result
            assert "erp_amplitude" in result
            assert "ignition_rate" in result

    def test_history_recording(self):
        """Test history recording."""
        extractor = NeuralObservableExtractor()

        # Take 50 steps
        for t in range(50):
            extractor.step(0.5, 1.0, 0)

        history = extractor.get_history()

        assert len(history["S"]) == 50
        assert len(history["gamma_power"]) == 50


class TestBehavioralObservableExtractor:
    """Test behavioral observable extraction."""

    def test_initialization(self):
        """Test behavioral extractor initialization."""
        extractor = BehavioralObservableExtractor()
        assert len(extractor.history) == 6

    def test_rt_variability_extraction(self):
        """Test RT variability extraction."""
        extractor = BehavioralObservableExtractor()

        # Create synthetic threshold history with variability
        theta_history = np.random.normal(1.0, 0.1, 150)

        rt_variability = extractor.extract_rt_variability(theta_history)

        # Should be positive
        assert rt_variability >= 0

    def test_response_criterion_extraction(self):
        """Test response criterion extraction."""
        extractor = BehavioralObservableExtractor()

        # Create synthetic threshold history
        theta_history = np.ones(150) * 1.5

        criterion = extractor.extract_response_criterion(theta_history)

        # Should be close to 1.5
        assert 1.4 < criterion < 1.6

    def test_decision_rate_extraction(self):
        """Test decision rate extraction."""
        extractor = BehavioralObservableExtractor()

        # Create synthetic ignition history
        B_history = np.zeros(150)
        B_history[::5] = 1  # 20% decision rate

        decision_rate = extractor.extract_decision_rate(B_history)

        # Should be close to 0.2
        assert 0.15 < decision_rate < 0.25

    def test_step(self):
        """Test step update."""
        extractor = BehavioralObservableExtractor()

        # Take multiple steps
        for t in range(100):
            S = np.sin(0.1 * t)
            theta = 1.0 + 0.1 * np.cos(0.05 * t)
            B = 1 if S > theta else 0

            result = extractor.step(S, theta, B)

            assert "rt_variability" in result
            assert "response_criterion" in result
            assert "decision_rate" in result


class TestKeyTestablePredictionValidator:
    """Test key testable prediction validation."""

    def test_initialization(self):
        """Test validator initialization."""
        validator = KeyTestablePredictionValidator(tau_sigma=0.5)
        assert validator.tau_sigma == 0.5

    def test_step(self):
        """Test validator step."""
        validator = KeyTestablePredictionValidator()

        result = validator.step(S=1.5, theta=1.0, B=1)

        assert "delta" in result
        assert "p_ign" in result
        assert result["delta"] == 0.5

    def test_margin_computation(self):
        """Test margin computation."""
        validator = KeyTestablePredictionValidator()

        # S > theta → positive margin
        result = validator.step(S=2.0, theta=1.0, B=1)
        assert result["delta"] > 0

        # S < theta → negative margin
        result = validator.step(S=0.5, theta=1.0, B=0)
        assert result["delta"] < 0

    def test_soft_ignition_probability(self):
        """Test soft ignition probability."""
        validator = KeyTestablePredictionValidator(tau_sigma=1.0)

        # Large positive margin → high p_ign
        result = validator.step(S=10.0, theta=0.0, B=1)
        assert result["p_ign"] > 0.9

        # Large negative margin → low p_ign
        result = validator.step(S=-10.0, theta=0.0, B=0)
        assert result["p_ign"] < 0.1

    def test_validation(self):
        """Test prediction validation."""
        validator = KeyTestablePredictionValidator()

        # Generate synthetic data
        np.random.seed(42)
        for t in range(200):
            S = np.sin(0.1 * t) + np.random.normal(0, 0.1)
            theta = 1.0 + 0.1 * np.cos(0.05 * t)
            B = 1 if S > theta else 0

            validator.step(S, theta, B)

        # Validate
        result = validator.validate()

        assert result["valid"]
        assert "correlation_margin" in result
        assert "correlation_signal" in result
        assert "margin_better" in result
        assert "improvement" in result

    def test_insufficient_data(self):
        """Test validation with insufficient data."""
        validator = KeyTestablePredictionValidator()

        # Only 10 samples
        for t in range(10):
            validator.step(0.5, 1.0, 0)

        result = validator.validate()

        assert not result["valid"]
        assert "Insufficient data" in result["reason"]


class TestParameterIdentifiabilityAnalyzer:
    """Test parameter identifiability analysis."""

    def test_fisher_information(self):
        """Test Fisher information computation."""
        # Generate synthetic data
        np.random.seed(42)
        S_history = np.random.normal(1.0, 0.5, 200)
        theta_history = np.random.normal(1.0, 0.2, 200)
        B_history = np.random.binomial(1, 0.3, 200)

        config = {
            "lam": 0.2,
            "eta": 0.1,
            "ignite_tau": 0.5,
        }

        result = ParameterIdentifiabilityAnalyzer.compute_fisher_information(
            S_history, theta_history, B_history, config
        )

        assert "fisher_information" in result
        assert "condition_number" in result
        assert "crlb_diag" in result
        assert "identifiable" in result

    def test_identifiability_constraints(self):
        """Test identifiability constraint checking."""
        config = {
            "lam": 0.2,
            "tau_s": 5.0,
            "eta": 0.1,
            "delta": 0.5,
            "ignite_tau": 0.5,
        }

        result = ParameterIdentifiabilityAnalyzer.check_identifiability_constraints(config)

        assert "constraint1_lam_tau_s_distinct" in result
        assert "constraint2_eta_delta_distinct" in result
        assert "constraint3_tau_sigma_positive" in result
        assert "all_satisfied" in result

    def test_constraint_violations(self):
        """Test detection of constraint violations."""
        # Violate constraint 3
        config = {
            "lam": 0.2,
            "tau_s": 5.0,
            "eta": 0.1,
            "delta": 0.5,
            "ignite_tau": -0.5,  # Negative!
        }

        result = ParameterIdentifiabilityAnalyzer.check_identifiability_constraints(config)

        assert not result["constraint3_tau_sigma_positive"]
        assert not result["all_satisfied"]


class TestObservableMappingIntegration:
    """Integration tests for observable mapping."""

    def test_full_pipeline(self):
        """Test full observable extraction pipeline."""
        neural = NeuralObservableExtractor()
        behavioral = BehavioralObservableExtractor()
        prediction = KeyTestablePredictionValidator()

        # Generate synthetic APGI dynamics
        np.random.seed(42)
        for t in range(500):
            S = np.sin(0.05 * t) + np.random.normal(0, 0.1)
            theta = 1.0 + 0.1 * np.cos(0.02 * t)
            B = 1 if S > theta else 0

            neural.step(S, theta, B)
            behavioral.step(S, theta, B)
            prediction.step(S, theta, B)

        # Check that all extractors have data
        neural_hist = neural.get_history()
        behavioral_hist = behavioral.get_history()
        prediction_hist = prediction.get_history()

        assert len(neural_hist["S"]) == 500
        assert len(behavioral_hist["S"]) == 500
        assert len(prediction_hist["S"]) == 500

        # Validate prediction
        pred_result = prediction.validate()
        assert pred_result["valid"]

    def test_neural_observable_extractor_clear_history(self):
        """Test clearing the neural observable extractor history."""
        extractor = NeuralObservableExtractor(fs=100.0)
        extractor.step(S=1.0, theta=0.5, B=0)
        assert len(extractor.history["S"]) == 1

        # Clear history manually
        extractor.history["S"] = []
        extractor.history["theta"] = []
        extractor.history["B"] = []
        assert len(extractor.history["S"]) == 0
        assert len(extractor.history["theta"]) == 0
        assert len(extractor.history["B"]) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])  # pragma: no cover
