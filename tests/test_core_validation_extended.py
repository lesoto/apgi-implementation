"""Extended tests for core/validation.py to achieve 100% coverage."""

import pytest
from core.validation import (
    validate_parameter,
    validate_config,
    ValidationError,
    get_constraint_summary,
    print_constraint_summary,
)


class TestValidateParameterExtended:
    """Extended tests for validate_parameter function."""

    def test_validate_parameter_in_range_pass(self):
        """Should pass when value is in range."""
        validate_parameter("lam", 0.5, "in (0, 1)", "§3.2")  # Should not raise

    def test_validate_parameter_in_range_fail(self):
        """Should raise when value is not in range."""
        with pytest.raises(ValidationError):
            validate_parameter("lam", 1.5, "in (0, 1)", "§3.2")

    def test_validate_parameter_greater_equal_pass(self):
        """Should pass for >= constraint."""
        validate_parameter("noise_std", 0.0, ">= 0", "§7.2")  # Should not raise

    def test_validate_parameter_greater_equal_fail(self):
        """Should raise for violated >= constraint."""
        with pytest.raises(ValidationError):
            validate_parameter("noise_std", -0.1, ">= 0", "§7.2")

    def test_validate_parameter_less_equal_pass(self):
        """Should pass for <= constraint."""
        validate_parameter("eta", 0.5, "<= 1", "§4.1")  # Should not raise

    def test_validate_parameter_less_equal_fail(self):
        """Should raise for violated <= constraint."""
        with pytest.raises(ValidationError):
            validate_parameter("eta", 1.5, "<= 1", "§4.1")

    def test_validate_parameter_less_than_pass(self):
        """Should pass for < constraint."""
        validate_parameter("lam", 0.9, "< 1", "§3.2")  # Should not raise

    def test_validate_parameter_less_than_fail(self):
        """Should raise for violated < constraint."""
        with pytest.raises(ValidationError):
            validate_parameter("lam", 1.0, "< 1", "§3.2")


class TestGetConstraintSummary:
    """Tests for get_constraint_summary function."""

    def test_summary_structure(self):
        """Should return structured summary."""
        summary = get_constraint_summary()
        assert isinstance(summary, dict)
        # Check expected categories
        expected_categories = [
            "Neuromodulator Separation",
            "Signal Accumulation",
            "Threshold Dynamics",
            "Ignition Mechanism",
            "Continuous-Time SDE",
            "Hierarchical Architecture",
            "Precision System",
            "Numerical Stability",
        ]
        for category in expected_categories:
            assert category in summary
            assert isinstance(summary[category], list)
            assert len(summary[category]) > 0


class TestPrintConstraintSummary:
    """Tests for print_constraint_summary function."""

    def test_prints_output(self, capsys):
        """Should print formatted summary."""
        print_constraint_summary()
        captured = capsys.readouterr()
        assert "APGI PARAMETER CONSTRAINTS" in captured.out
        assert "Neuromodulator Separation" in captured.out
        assert "=" in captured.out


class TestValidateConfigExtended:
    """Extended tests for validate_config function."""

    def test_validate_config_timescale_k_invalid(self):
        """Should raise ValidationError when timescale_k <= 1."""
        config = {"use_hierarchical": True, "timescale_k": 1.0}
        with pytest.raises(ValidationError):
            validate_config(config)

    def test_validate_config_pi_min_invalid(self):
        """Should raise ValidationError when pi_min <= 0."""
        config = {"pi_min": 0.0}
        with pytest.raises(ValidationError):
            validate_config(config)

    def test_validate_config_pi_max_invalid(self):
        """Should raise ValidationError when pi_max <= pi_min."""
        config = {"pi_min": 1.0, "pi_max": 0.5}
        with pytest.raises(ValidationError):
            validate_config(config)

    def test_validate_config_eps_invalid(self):
        """Should raise ValidationError when eps not in (0, 1)."""
        config = {"eps": 0.0}
        with pytest.raises(ValidationError):
            validate_config(config)

        config = {"eps": 1.0}
        with pytest.raises(ValidationError):
            validate_config(config)

    def test_validate_config_eta_invalid(self):
        """Should raise ValidationError when eta not in (0, 1]."""
        config = {"eta": 0.0}
        with pytest.raises(ValidationError):
            validate_config(config)

        config = {"eta": 1.5}
        with pytest.raises(ValidationError):
            validate_config(config)

    def test_validate_config_noise_std_invalid(self):
        """Should raise ValidationError when noise_std < 0."""
        config = {"noise_std": -0.1}
        with pytest.raises(ValidationError):
            validate_config(config)
