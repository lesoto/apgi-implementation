"""Tests for validation/observable_mapping.py neural and behavioral observable extraction."""

import numpy as np

from validation.observable_mapping import (
    BehavioralObservableExtractor,
    KeyTestablePredictionValidator,
    NeuralObservableExtractor,
    ParameterIdentifiabilityAnalyzer,
)


class TestNeuralObservableExtractor:
    """Test neural observable extraction."""

    def test_initialization(self):
        """Test NeuralObservableExtractor initialization."""
        extractor = NeuralObservableExtractor(fs=100.0)
        assert extractor.fs == 100.0
        assert "S" in extractor.history
        assert "theta" in extractor.history
        assert "B" in extractor.history
        assert "gamma_power" in extractor.history
        assert "erp_amplitude" in extractor.history
        assert "ignition_rate" in extractor.history

    def test_extract_gamma_power_insufficient_data(self):
        """Test gamma power extraction with insufficient data."""
        extractor = NeuralObservableExtractor(fs=100.0)
        signal = np.array([1.0, 2.0, 3.0])
        power = extractor.extract_gamma_power(signal)
        assert power == 0.0

    def test_extract_gamma_power_sufficient_data(self):
        """Test gamma power extraction with sufficient data."""
        extractor = NeuralObservableExtractor(fs=100.0)
        signal = np.random.randn(100)
        power = extractor.extract_gamma_power(signal)
        assert isinstance(power, float)
        assert power >= 0.0

    def test_extract_gamma_power_custom_freq_range(self):
        """Test gamma power extraction with custom frequency range."""
        extractor = NeuralObservableExtractor(fs=100.0)
        signal = np.random.randn(100)
        power = extractor.extract_gamma_power(signal, freq_range=(20, 50))
        assert isinstance(power, float)
        assert power >= 0.0

    def test_extract_erp_amplitude_insufficient_data(self):
        """Test ERP amplitude with insufficient data."""
        extractor = NeuralObservableExtractor(fs=100.0)
        theta_history = np.array([1.0, 2.0])
        amplitude = extractor.extract_erp_amplitude(theta_history, window_size=50)
        assert isinstance(amplitude, float)

    def test_extract_erp_amplitude_empty_history(self):
        """Test ERP amplitude with empty history."""
        extractor = NeuralObservableExtractor(fs=100.0)
        theta_history = np.array([])
        amplitude = extractor.extract_erp_amplitude(theta_history, window_size=50)
        assert amplitude == 0.0

    def test_extract_erp_amplitude_sufficient_data(self):
        """Test ERP amplitude with sufficient data."""
        extractor = NeuralObservableExtractor(fs=100.0)
        theta_history = np.concatenate([np.ones(50) * 0.5, np.ones(50) * 1.5])
        amplitude = extractor.extract_erp_amplitude(theta_history, window_size=50)
        assert isinstance(amplitude, float)
        # Should be positive since peak > baseline
        assert amplitude > 0

    def test_extract_ignition_rate_insufficient_data(self):
        """Test ignition rate with insufficient data."""
        extractor = NeuralObservableExtractor(fs=100.0)
        B_history = np.array([0, 1, 0])
        rate = extractor.extract_ignition_rate(B_history, window_size=100)
        assert isinstance(rate, float)

    def test_extract_ignition_rate_empty_history(self):
        """Test ignition rate with empty history."""
        extractor = NeuralObservableExtractor(fs=100.0)
        B_history = np.array([])
        rate = extractor.extract_ignition_rate(B_history, window_size=100)
        assert rate == 0.0

    def test_extract_ignition_rate_sufficient_data(self):
        """Test ignition rate with sufficient data."""
        extractor = NeuralObservableExtractor(fs=100.0)
        B_history = np.array([0] * 50 + [1] * 50)
        rate = extractor.extract_ignition_rate(B_history, window_size=100)
        assert isinstance(rate, float)
        assert rate == 0.5

    def test_step_updates_history(self):
        """Test that step updates history correctly."""
        extractor = NeuralObservableExtractor(fs=100.0)
        result = extractor.step(S=1.0, theta=0.5, B=0)
        assert "gamma_power" in result
        assert "erp_amplitude" in result
        assert "ignition_rate" in result
        assert len(extractor.history["S"]) == 1
        assert len(extractor.history["theta"]) == 1
        assert len(extractor.history["B"]) == 1

    def test_step_multiple_calls(self):
        """Test multiple step calls accumulate history."""
        extractor = NeuralObservableExtractor(fs=100.0)
        for i in range(10):
            extractor.step(S=float(i), theta=0.5, B=0)
        assert len(extractor.history["S"]) == 10
        assert len(extractor.history["theta"]) == 10
        assert len(extractor.history["B"]) == 10

    def test_get_history(self):
        """Test get_history returns copy."""
        extractor = NeuralObservableExtractor(fs=100.0)
        extractor.step(S=1.0, theta=0.5, B=0)
        history = extractor.get_history()
        assert history == extractor.history
        # Should be a copy, not the same object
        assert history is not extractor.history


class TestBehavioralObservableExtractor:
    """Test behavioral observable extraction."""

    def test_initialization(self):
        """Test BehavioralObservableExtractor initialization."""
        extractor = BehavioralObservableExtractor()
        assert "S" in extractor.history
        assert "theta" in extractor.history
        assert "B" in extractor.history
        assert "rt_variability" in extractor.history
        assert "response_criterion" in extractor.history
        assert "decision_rate" in extractor.history

    def test_extract_rt_variability_insufficient_data(self):
        """Test RT variability with insufficient data."""
        extractor = BehavioralObservableExtractor()
        theta_history = np.array([1.0, 2.0])
        variability = extractor.extract_rt_variability(theta_history, window_size=100)
        assert variability == 0.0

    def test_extract_rt_variability_sufficient_data(self):
        """Test RT variability with sufficient data."""
        extractor = BehavioralObservableExtractor()
        theta_history = np.random.randn(150)
        variability = extractor.extract_rt_variability(theta_history, window_size=100)
        assert isinstance(variability, float)
        assert variability >= 0.0

    def test_extract_response_criterion_insufficient_data(self):
        """Test response criterion with insufficient data."""
        extractor = BehavioralObservableExtractor()
        theta_history = np.array([1.0, 2.0])
        criterion = extractor.extract_response_criterion(theta_history, window_size=100)
        assert isinstance(criterion, float)

    def test_extract_response_criterion_empty_history(self):
        """Test response criterion with empty history."""
        extractor = BehavioralObservableExtractor()
        theta_history = np.array([])
        criterion = extractor.extract_response_criterion(theta_history, window_size=100)
        assert criterion == 0.0

    def test_extract_response_criterion_sufficient_data(self):
        """Test response criterion with sufficient data."""
        extractor = BehavioralObservableExtractor()
        theta_history = np.ones(150) * 0.7
        criterion = extractor.extract_response_criterion(theta_history, window_size=100)
        assert isinstance(criterion, float)
        assert abs(criterion - 0.7) < 1e-6

    def test_extract_decision_rate_insufficient_data(self):
        """Test decision rate with insufficient data."""
        extractor = BehavioralObservableExtractor()
        B_history = np.array([0, 1, 0])
        rate = extractor.extract_decision_rate(B_history, window_size=100)
        assert isinstance(rate, float)

    def test_extract_decision_rate_empty_history(self):
        """Test decision rate with empty history."""
        extractor = BehavioralObservableExtractor()
        B_history = np.array([])
        rate = extractor.extract_decision_rate(B_history, window_size=100)
        assert rate == 0.0

    def test_extract_decision_rate_sufficient_data(self):
        """Test decision rate with sufficient data."""
        extractor = BehavioralObservableExtractor()
        B_history = np.array([0] * 50 + [1] * 50)
        rate = extractor.extract_decision_rate(B_history, window_size=100)
        assert isinstance(rate, float)
        assert rate == 0.5

    def test_step_updates_history(self):
        """Test that step updates history correctly."""
        extractor = BehavioralObservableExtractor()
        result = extractor.step(S=1.0, theta=0.5, B=0)
        assert "rt_variability" in result
        assert "response_criterion" in result
        assert "decision_rate" in result
        assert len(extractor.history["S"]) == 1
        assert len(extractor.history["theta"]) == 1
        assert len(extractor.history["B"]) == 1

    def test_get_history(self):
        """Test get_history returns copy."""
        extractor = BehavioralObservableExtractor()
        extractor.step(S=1.0, theta=0.5, B=0)
        history = extractor.get_history()
        assert history == extractor.history
        assert history is not extractor.history


class TestKeyTestablePredictionValidator:
    """Test key prediction validation."""

    def test_initialization(self):
        """Test KeyTestablePredictionValidator initialization."""
        validator = KeyTestablePredictionValidator(tau_sigma=0.5)
        assert validator.tau_sigma == 0.5
        assert "S" in validator.history
        assert "theta" in validator.history
        assert "B" in validator.history
        assert "delta" in validator.history
        assert "p_ign" in validator.history

    def test_step_computes_margin(self):
        """Test that step computes ignition margin."""
        validator = KeyTestablePredictionValidator(tau_sigma=0.5)
        result = validator.step(S=1.0, theta=0.5, B=0)
        assert "delta" in result
        assert "p_ign" in result
        assert result["delta"] == 0.5  # 1.0 - 0.5
        assert 0.0 <= result["p_ign"] <= 1.0

    def test_step_sigmoid_probability(self):
        """Test sigmoid probability computation."""
        validator = KeyTestablePredictionValidator(tau_sigma=0.5)
        # Positive margin should give p_ign > 0.5
        result1 = validator.step(S=1.0, theta=0.0, B=0)
        assert result1["p_ign"] > 0.5
        # Negative margin should give p_ign < 0.5
        result2 = validator.step(S=0.0, theta=1.0, B=0)
        assert result2["p_ign"] < 0.5

    def test_validate_insufficient_data(self):
        """Test validation with insufficient data."""
        validator = KeyTestablePredictionValidator(tau_sigma=0.5)
        validator.step(S=1.0, theta=0.5, B=0)
        result = validator.validate()
        assert result["valid"] is False
        assert "reason" in result

    def test_validate_sufficient_data(self):
        """Test validation with sufficient data."""
        validator = KeyTestablePredictionValidator(tau_sigma=0.5)
        # Add 100 samples
        for i in range(100):
            B = 1 if i % 10 == 0 else 0
            validator.step(S=float(i) / 100.0, theta=0.5, B=B)
        result = validator.validate()
        assert result["valid"] is True
        assert "correlation_margin" in result
        assert "correlation_signal" in result
        assert "margin_better" in result
        assert "improvement" in result

    def test_validate_correlations(self):
        """Test that validation computes correlations."""
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            validator = KeyTestablePredictionValidator(tau_sigma=0.5)
            # Create data where margin correlates with ignition
            for i in range(100):
                S = 0.8 + 0.1 * np.random.randn()
                theta = 0.5
                B = 1 if S > theta else 0
                validator.step(S=S, theta=theta, B=B)
            result = validator.validate()
            assert isinstance(result["correlation_margin"], float)
            assert isinstance(result["correlation_signal"], float)

    def test_get_history(self):
        """Test get_history returns copy."""
        validator = KeyTestablePredictionValidator(tau_sigma=0.5)
        validator.step(S=1.0, theta=0.5, B=0)
        history = validator.get_history()
        assert history == validator.history
        assert history is not validator.history


class TestParameterIdentifiabilityAnalyzer:
    """Test parameter identifiability analysis."""

    def test_compute_fisher_information_basic(self):
        """Test Fisher information computation."""
        S_history = np.random.randn(100)
        theta_history = np.random.randn(100)
        B_history = np.random.randint(0, 2, 100)
        params = {"lam": 0.2, "eta": 0.1, "ignite_tau": 0.5}
        result = ParameterIdentifiabilityAnalyzer.compute_fisher_information(
            S_history, theta_history, B_history, params
        )
        assert "fisher_information" in result
        assert "condition_number" in result
        assert "crlb_diag" in result
        assert "identifiable" in result

    def test_fisher_information_matrix_shape(self):
        """Test Fisher information matrix has correct shape."""
        S_history = np.random.randn(50)
        theta_history = np.random.randn(50)
        B_history = np.random.randint(0, 2, 50)
        params = {"lam": 0.2}
        result = ParameterIdentifiabilityAnalyzer.compute_fisher_information(
            S_history, theta_history, B_history, params
        )
        assert result["fisher_information"].shape == (3, 3)

    def test_check_identifiability_constraints_all_satisfied(self):
        """Test identifiability constraints when all satisfied."""
        config = {
            "lam": 0.15,  # Different from 1/tau_s = 0.2
            "tau_s": 5.0,
            "eta": 0.1,
            "delta": 0.5,  # Different from eta
            "ignite_tau": 0.5,
        }
        result = ParameterIdentifiabilityAnalyzer.check_identifiability_constraints(
            config
        )
        assert result["constraint1_lam_tau_s_distinct"] is True
        assert result["constraint2_eta_delta_distinct"] is True
        assert result["constraint3_tau_sigma_positive"] is True
        assert result["all_satisfied"] is True

    def test_check_identifiability_constraint1_violated(self):
        """Test constraint 1 violation (lam and tau_s not distinct)."""
        config = {"lam": 0.2, "tau_s": 5.0, "eta": 0.1, "delta": 0.5, "ignite_tau": 0.5}
        # lam = 0.2, 1/tau_s = 0.2, so they're not distinct
        result = ParameterIdentifiabilityAnalyzer.check_identifiability_constraints(
            config
        )
        assert result["constraint1_lam_tau_s_distinct"] is False
        assert result["all_satisfied"] is False

    def test_check_identifiability_constraint2_violated(self):
        """Test constraint 2 violation (eta and delta not distinct)."""
        config = {"lam": 0.2, "tau_s": 5.0, "eta": 0.5, "delta": 0.5, "ignite_tau": 0.5}
        result = ParameterIdentifiabilityAnalyzer.check_identifiability_constraints(
            config
        )
        assert result["constraint2_eta_delta_distinct"] is False
        assert result["all_satisfied"] is False

    def test_check_identifiability_constraint3_violated(self):
        """Test constraint 3 violation (tau_sigma not positive)."""
        config = {
            "lam": 0.2,
            "tau_s": 5.0,
            "eta": 0.1,
            "delta": 0.5,
            "ignite_tau": -0.5,
        }
        result = ParameterIdentifiabilityAnalyzer.check_identifiability_constraints(
            config
        )
        assert result["constraint3_tau_sigma_positive"] is False
        assert result["all_satisfied"] is False

    def test_check_identifiability_missing_params(self):
        """Test with missing parameters (uses defaults)."""
        config = {}  # Empty config
        result = ParameterIdentifiabilityAnalyzer.check_identifiability_constraints(
            config
        )
        # Should use defaults and check constraints
        assert "constraint1_lam_tau_s_distinct" in result
        assert "constraint2_eta_delta_distinct" in result
        assert "constraint3_tau_sigma_positive" in result
