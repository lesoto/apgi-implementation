"""Unit tests for parameter validation module.

Tests APGI Spec §15: Design Constraints
"""

import pytest

from core.validation import (
    ValidationError,
    get_constraint_summary,
    print_constraint_summary,
    validate_config,
    validate_parameter,
    validate_reset_factor,
)


class TestNeuromodulatorSeparation:
    """Test neuromodulator separation constraints (§2.3-2.4)."""

    def test_both_ne_modes_raises_error(self):
        """Should raise error when both NE modes enabled."""
        config = {
            "ne_on_precision": True,
            "ne_on_threshold": True,
        }
        with pytest.raises(ValidationError):
            validate_config(config)

    def test_ne_precision_only_passes(self):
        """Should pass with only NE on precision."""
        config = {
            "ne_on_precision": True,
            "ne_on_threshold": False,
            "lam": 0.2,
            "kappa": 0.15,
            "ignite_tau": 0.5,
            "dt": 0.1,  # Must be ≤ min(τ)/10 = 5.0/10 = 0.5
            "tau_s": 5.0,
            "tau_theta": 1000.0,
            "tau_pi": 1000.0,
            "pi_min": 1e-4,
            "pi_max": 1e4,
            "eps": 1e-8,
            "eta": 0.1,
            "noise_std": 0.01,
        }
        validate_config(config)  # Should not raise

    def test_ne_threshold_only_passes(self):
        """Should pass with only NE on threshold."""
        config = {
            "ne_on_precision": False,
            "ne_on_threshold": True,
            "lam": 0.2,
            "kappa": 0.15,
            "ignite_tau": 0.5,
            "dt": 0.1,  # Must be ≤ min(τ)/10 = 5.0/10 = 0.5
            "tau_s": 5.0,
            "tau_theta": 1000.0,
            "tau_pi": 1000.0,
            "pi_min": 1e-4,
            "pi_max": 1e4,
            "eps": 1e-8,
            "eta": 0.1,
            "noise_std": 0.01,
        }
        validate_config(config)  # Should not raise


class TestSignalAccumulation:
    """Test signal accumulation constraints (§3.2)."""

    def test_lam_zero_raises_error(self):
        """Should raise error when lam = 0."""
        config = {"lam": 0.0}
        with pytest.raises(ValidationError):
            validate_config(config)

    def test_lam_one_raises_error(self):
        """Should raise error when lam = 1."""
        config = {"lam": 1.0}
        with pytest.raises(ValidationError):
            validate_config(config)

    def test_lam_negative_raises_error(self):
        """Should raise error when lam < 0."""
        config = {"lam": -0.1}
        with pytest.raises(ValidationError):
            validate_config(config)

    def test_lam_valid_passes(self):
        """Should pass with valid lam."""
        config = {
            "lam": 0.2,
            "kappa": 0.15,
            "ignite_tau": 0.5,
            "dt": 0.1,  # Must be ≤ min(τ)/10
            "tau_s": 5.0,
            "tau_theta": 1000.0,
            "tau_pi": 1000.0,
            "pi_min": 1e-4,
            "pi_max": 1e4,
            "eps": 1e-8,
            "eta": 0.1,
            "noise_std": 0.01,
            "ne_on_precision": False,
            "ne_on_threshold": False,
        }
        validate_config(config)  # Should not raise


class TestThresholdDynamics:
    """Test threshold dynamics constraints (§4, §6)."""

    def test_kappa_zero_raises_error(self):
        """Should raise error when kappa = 0."""
        config = {"kappa": 0.0}
        with pytest.raises(ValidationError):
            validate_config(config)

    def test_kappa_negative_raises_error(self):
        """Should raise error when kappa < 0."""
        config = {"kappa": -0.1}
        with pytest.raises(ValidationError):
            validate_config(config)

    def test_kappa_valid_passes(self):
        """Should pass with valid kappa."""
        config = {
            "kappa": 0.15,
            "lam": 0.2,
            "ignite_tau": 0.5,
            "dt": 0.1,  # Must be ≤ min(τ)/10
            "tau_s": 5.0,
            "tau_theta": 1000.0,
            "tau_pi": 1000.0,
            "pi_min": 1e-4,
            "pi_max": 1e4,
            "eps": 1e-8,
            "eta": 0.1,
            "noise_std": 0.01,
            "ne_on_precision": False,
            "ne_on_threshold": False,
        }
        validate_config(config)  # Should not raise

    def test_reset_factor_zero_raises_error(self):
        """Should raise error when reset_factor = 0."""
        config = {"reset_factor": 0.0}
        with pytest.raises(ValidationError):
            validate_config(config)

    def test_reset_factor_one_raises_error(self):
        """Should raise error when reset_factor = 1."""
        config = {"reset_factor": 1.0}
        with pytest.raises(ValidationError):
            validate_config(config)

    def test_reset_factor_valid_passes(self):
        """Should pass with valid reset_factor."""
        config = {
            "reset_factor": 0.5,
            "kappa": 0.15,
            "lam": 0.2,
            "ignite_tau": 0.5,
            "dt": 0.1,  # Must be ≤ min(τ)/10
            "tau_s": 5.0,
            "tau_theta": 1000.0,
            "tau_pi": 1000.0,
            "pi_min": 1e-4,
            "pi_max": 1e4,
            "eps": 1e-8,
            "eta": 0.1,
            "noise_std": 0.01,
            "ne_on_precision": False,
            "ne_on_threshold": False,
        }
        validate_config(config)  # Should not raise


class TestIgnitionDynamics:
    """Test ignition mechanism constraints (§5.2)."""

    def test_ignite_tau_zero_raises_error(self):
        """Should raise error when ignite_tau = 0."""
        config = {"ignite_tau": 0.0}
        with pytest.raises(ValidationError):
            validate_config(config)

    def test_ignite_tau_negative_raises_error(self):
        """Should raise error when ignite_tau < 0."""
        config = {"ignite_tau": -0.1}
        with pytest.raises(ValidationError):
            validate_config(config)

    def test_ignite_tau_valid_passes(self):
        """Should pass with valid ignite_tau."""
        config = {
            "ignite_tau": 0.5,
            "kappa": 0.15,
            "lam": 0.2,
            "dt": 0.1,  # Must be ≤ min(τ)/10
            "tau_s": 5.0,
            "tau_theta": 1000.0,
            "tau_pi": 1000.0,
            "pi_min": 1e-4,
            "pi_max": 1e4,
            "eps": 1e-8,
            "eta": 0.1,
            "noise_std": 0.01,
            "ne_on_precision": False,
            "ne_on_threshold": False,
        }
        validate_config(config)  # Should not raise


class TestContinuousTimeSDE:
    """Test continuous-time SDE constraints (§7.4)."""

    def test_dt_zero_raises_error(self):
        """Should raise error when dt = 0."""
        config = {"dt": 0.0}
        with pytest.raises(ValidationError):
            validate_config(config)

    def test_dt_negative_raises_error(self):
        """Should raise error when dt < 0."""
        config = {"dt": -0.1}
        with pytest.raises(ValidationError):
            validate_config(config)

    def test_dt_too_large_raises_error(self):
        """Should raise error when dt > min(τ)/10."""
        config = {
            "dt": 1000.0,  # Very large
            "tau_s": 5.0,
            "tau_theta": 1000.0,
            "tau_pi": 1000.0,
        }
        with pytest.raises(ValidationError):
            validate_config(config)

    def test_dt_valid_passes(self):
        """Should pass with valid dt."""
        config = {
            "dt": 0.1,  # Small enough
            "tau_s": 5.0,
            "tau_theta": 1000.0,
            "tau_pi": 1000.0,
            "kappa": 0.15,
            "lam": 0.2,
            "ignite_tau": 0.5,
            "pi_min": 1e-4,
            "pi_max": 1e4,
            "eps": 1e-8,
            "eta": 0.1,
            "noise_std": 0.01,
            "ne_on_precision": False,
            "ne_on_threshold": False,
        }
        validate_config(config)  # Should not raise


class TestHierarchicalParameters:
    """Test hierarchical architecture constraints (§8.1)."""

    def test_timescale_k_one_raises_error(self):
        """Should raise error when timescale_k = 1."""
        config = {
            "use_hierarchical": True,
            "timescale_k": 1.0,
        }
        with pytest.raises(ValidationError):
            validate_config(config)

    def test_timescale_k_less_than_one_raises_error(self):
        """Should raise error when timescale_k < 1."""
        config = {
            "use_hierarchical": True,
            "timescale_k": 0.5,
        }
        with pytest.raises(ValidationError):
            validate_config(config)

    def test_timescale_k_valid_passes(self):
        """Should pass with valid timescale_k."""
        config = {
            "use_hierarchical": True,
            "timescale_k": 1.6,
            "tau_0": 2.0,  # Must be > 1
            "n_levels": 3,
            "kappa": 0.15,
            "lam": 0.2,
            "ignite_tau": 0.5,
            "dt": 0.1,  # Must be ≤ min(τ)/10
            "tau_s": 5.0,
            "tau_theta": 1000.0,
            "tau_pi": 1000.0,
            "pi_min": 1e-4,
            "pi_max": 1e4,
            "eps": 1e-8,
            "eta": 0.1,
            "noise_std": 0.01,
            "ne_on_precision": False,
            "ne_on_threshold": False,
        }
        validate_config(config)  # Should not raise


class TestPrecisionParameters:
    """Test precision system constraints (§2.2)."""

    def test_pi_min_zero_raises_error(self):
        """Should raise error when pi_min = 0."""
        config = {"pi_min": 0.0}
        with pytest.raises(ValidationError):
            validate_config(config)

    def test_pi_min_negative_raises_error(self):
        """Should raise error when pi_min < 0."""
        config = {"pi_min": -0.01}
        with pytest.raises(ValidationError):
            validate_config(config)

    def test_pi_max_less_than_pi_min_raises_error(self):
        """Should raise error when pi_max ≤ pi_min."""
        config = {
            "pi_min": 1e-4,
            "pi_max": 1e-5,
        }
        with pytest.raises(ValidationError):
            validate_config(config)

    def test_pi_bounds_valid_passes(self):
        """Should pass with valid precision bounds."""
        config = {
            "pi_min": 1e-4,
            "pi_max": 1e4,
            "kappa": 0.15,
            "lam": 0.2,
            "ignite_tau": 0.5,
            "dt": 0.1,
            "tau_s": 5.0,
            "tau_theta": 1000.0,
            "tau_pi": 1000.0,
            "eps": 1e-8,
            "eta": 0.1,
            "noise_std": 0.01,
            "ne_on_precision": False,
            "ne_on_threshold": False,
        }
        validate_config(config)  # Should not raise


class TestNumericalStability:
    """Test general numerical stability constraints."""

    def test_eps_zero_raises_error(self):
        """Should raise error when eps = 0."""
        config = {"eps": 0.0}
        with pytest.raises(ValidationError):
            validate_config(config)

    def test_eps_one_raises_error(self):
        """Should raise error when eps = 1."""
        config = {"eps": 1.0}
        with pytest.raises(ValidationError):
            validate_config(config)

    def test_eta_zero_raises_error(self):
        """Should raise error when eta = 0."""
        config = {"eta": 0.0}
        with pytest.raises(ValidationError):
            validate_config(config)

    def test_eta_greater_than_one_raises_error(self):
        """Should raise error when eta > 1."""
        config = {"eta": 1.5}
        with pytest.raises(ValidationError):
            validate_config(config)

    def test_noise_std_negative_raises_error(self):
        """Should raise error when noise_std < 0."""
        config = {"noise_std": -0.01}
        with pytest.raises(ValidationError):
            validate_config(config)


class TestValidateParameter:
    """Test single parameter validation."""

    def test_validate_greater_than(self):
        """Should validate > constraint."""
        validate_parameter("kappa", 0.15, "> 0", "§4.5")
        with pytest.raises(ValidationError):
            validate_parameter("kappa", 0.0, "> 0", "§4.5")

    def test_validate_in_range(self):
        """Should validate in (a, b) constraint."""
        validate_parameter("lam", 0.2, "in (0, 1)", "§3.2")
        with pytest.raises(ValidationError):
            validate_parameter("lam", 0.0, "in (0, 1)", "§3.2")
        with pytest.raises(ValidationError):
            validate_parameter("lam", 1.0, "in (0, 1)", "§3.2")

    def test_validate_less_than(self):
        """Should validate < constraint."""
        validate_parameter("lam", 0.5, "< 1", "§3.2")
        with pytest.raises(ValidationError):
            validate_parameter("lam", 1.0, "< 1", "§3.2")


class TestConstraintSummary:
    """Test constraint summary generation."""

    def test_summary_structure(self):
        """Summary should have expected structure."""
        summary = get_constraint_summary()
        assert isinstance(summary, dict)
        assert len(summary) > 0

    def test_summary_categories(self):
        """Summary should have expected categories."""
        summary = get_constraint_summary()
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

    def test_summary_constraints(self):
        """Summary should list constraints."""
        summary = get_constraint_summary()
        for category, constraints in summary.items():
            assert isinstance(constraints, list)
            assert len(constraints) > 0
            for constraint in constraints:
                assert isinstance(constraint, str)


class TestValidateResetFactor:
    """Test standalone reset factor validation function."""

    def test_valid_reset_factor(self):
        """Should pass with valid reset_factor."""
        validate_reset_factor(0.5)  # Should not raise
        validate_reset_factor(0.1)  # Should not raise
        validate_reset_factor(0.9)  # Should not raise

    def test_reset_factor_zero_raises(self):
        """Should raise when reset_factor = 0."""
        with pytest.raises(ValidationError, match="reset_factor must be in"):
            validate_reset_factor(0.0)

    def test_reset_factor_one_raises(self):
        """Should raise when reset_factor = 1."""
        with pytest.raises(ValidationError, match="reset_factor must be in"):
            validate_reset_factor(1.0)

    def test_reset_factor_negative_raises(self):
        """Should raise when reset_factor < 0."""
        with pytest.raises(ValidationError, match="reset_factor must be in"):
            validate_reset_factor(-0.1)

    def test_reset_factor_greater_than_one_raises(self):
        """Should raise when reset_factor > 1."""
        with pytest.raises(ValidationError, match="reset_factor must be in"):
            validate_reset_factor(1.5)


class TestValidateParameterExtended:
    """Extended tests for validate_parameter with all constraint types."""

    def test_validate_greater_equal(self):
        """Should validate >= constraint."""
        validate_parameter("noise_std", 0.0, ">= 0", "§7.2")
        validate_parameter("noise_std", 0.01, ">= 0", "§7.2")
        with pytest.raises(ValidationError):
            validate_parameter("noise_std", -0.01, ">= 0", "§7.2")

    def test_validate_less_equal(self):
        """Should validate <= constraint."""
        validate_parameter("eta", 0.5, "<= 1", "§4.1")
        validate_parameter("eta", 1.0, "<= 1", "§4.1")
        with pytest.raises(ValidationError):
            validate_parameter("eta", 1.5, "<= 1", "§4.1")

    def test_validate_in_range_open(self):
        """Should validate in (a, b) with open bounds."""
        validate_parameter("lam", 0.5, "in (0, 1)", "§3.2")
        with pytest.raises(ValidationError):
            validate_parameter("lam", 0.0, "in (0, 1)", "§3.2")
        with pytest.raises(ValidationError):
            validate_parameter("lam", 1.0, "in (0, 1)", "§3.2")


class TestPrecisionWarning:
    """Test precision parameter warnings."""

    def test_very_large_precision_range_warns(self):
        """Should warn when precision range is very large."""
        config = {
            "pi_min": 1e-10,
            "pi_max": 1e10,
            "kappa": 0.15,
            "lam": 0.2,
            "ignite_tau": 0.5,
            "dt": 0.1,
            "tau_s": 5.0,
            "tau_theta": 1000.0,
            "tau_pi": 1000.0,
            "eps": 1e-8,
            "eta": 0.1,
            "noise_std": 0.01,
            "ne_on_precision": False,
            "ne_on_threshold": False,
        }
        with pytest.warns(RuntimeWarning, match="Precision range very large"):
            validate_config(config)


class TestTimescaleValidation:
    """Test hierarchical timescale validation."""

    def test_tau_0_less_than_one_raises_error(self):
        """Should raise error when tau_0 results in tau_ell <= 1."""
        config = {
            "use_hierarchical": True,
            "timescale_k": 1.6,
            "tau_0": 0.5,  # Too small - will make tau_ell <= 1
            "n_levels": 3,
            "kappa": 0.15,
            "lam": 0.2,
            "ignite_tau": 0.5,
            "dt": 0.1,
            "tau_s": 5.0,
            "tau_theta": 1000.0,
            "tau_pi": 1000.0,
            "pi_min": 1e-4,
            "pi_max": 1e4,
            "eps": 1e-8,
            "eta": 0.1,
            "noise_std": 0.01,
            "ne_on_precision": False,
            "ne_on_threshold": False,
        }
        with pytest.raises(ValidationError, match="τ_.*≤ 1"):
            validate_config(config)

    def test_valid_hierarchical_config_passes(self):
        """Should pass with valid hierarchical config."""
        config = {
            "use_hierarchical": True,
            "timescale_k": 1.6,
            "tau_0": 10.0,
            "n_levels": 3,
            "kappa": 0.15,
            "lam": 0.2,
            "ignite_tau": 0.5,
            "dt": 0.1,
            "tau_s": 5.0,
            "tau_theta": 1000.0,
            "tau_pi": 1000.0,
            "pi_min": 1e-4,
            "pi_max": 1e4,
            "eps": 1e-8,
            "eta": 0.1,
            "noise_std": 0.01,
            "ne_on_precision": False,
            "ne_on_threshold": False,
        }
        validate_config(config)  # Should not raise


class TestPrintConstraintSummary:
    """Test print_constraint_summary function."""

    def test_prints_output(self, capsys):
        """Should print constraint summary to stdout."""
        print_constraint_summary()
        captured = capsys.readouterr()
        assert "APGI PARAMETER CONSTRAINTS" in captured.out
        assert "Neuromodulator Separation" in captured.out
        assert "Signal Accumulation" in captured.out

    def test_prints_all_categories(self, capsys):
        """Should print all constraint categories."""
        print_constraint_summary()
        captured = capsys.readouterr()
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
            assert category in captured.out


class TestConstraintSummaryEdgeCases:
    """Test constraint summary edge cases."""

    def test_summary_returns_dict(self):
        """get_constraint_summary should return a dictionary."""
        summary = get_constraint_summary()
        assert isinstance(summary, dict)

    def test_summary_has_values(self):
        """Summary should have values for all keys."""
        summary = get_constraint_summary()
        for category, constraints in summary.items():
            assert isinstance(category, str)
            assert len(category) > 0
            assert isinstance(constraints, list)
            assert len(constraints) > 0

    def test_constraint_summary_empty_config(self):
        """Test constraint summary with empty configuration."""
        summary = get_constraint_summary()
        # Should still return a valid summary structure
        assert isinstance(summary, dict)
        assert len(summary) > 0

    def test_constraint_summary_structure(self):
        """Test constraint summary returns correct structure."""
        summary = get_constraint_summary()
        # Should have categories
        assert isinstance(summary, dict)
        # Each category should have constraints
        for category, constraints in summary.items():
            assert isinstance(constraints, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])  # pragma: no cover
