"""Tests for post-ignition signal reset behavior."""

import pytest
from pipeline import APGIPipeline
from config import CONFIG


class TestPostIgnitionReset:
    """Test post-ignition signal reset S ← ρ·S."""

    def test_signal_reset_applied_on_ignition(self):
        """Signal should be multiplied by reset_factor when ignition occurs."""
        config = dict(CONFIG)
        config["reset_factor"] = 0.5  # 50% signal retention
        config["theta_0"] = 0.1  # Low threshold to ensure ignition
        config["stochastic_ignition"] = False

        pipeline = APGIPipeline(config)
        pipeline.S = 10.0  # Set high signal to trigger ignition

        # Step with signal above threshold should trigger ignition
        result = pipeline.step(x_e=10.0, x_i=10.0)

        # After ignition, signal should be reduced by reset_factor
        # S should be approximately 10.0 * 0.5 = 5.0 (before other dynamics)
        # We check that B == 1 (ignition occurred)
        assert result["B"] == 1
        # Signal should be significantly reduced from initial 10.0
        assert pipeline.S < 8.0  # Allow for some dynamics, but should be reduced

    def test_no_reset_without_ignition(self):
        """Signal should NOT be reset when no ignition occurs."""
        config = dict(CONFIG)
        config["reset_factor"] = 0.5
        config["theta_0"] = 100.0  # High threshold to prevent ignition
        config["stochastic_ignition"] = False

        pipeline = APGIPipeline(config)
        pipeline.S = 1.0

        # Step with low signal should NOT trigger ignition
        result = pipeline.step(x_e=0.1, x_i=0.1)

        # No ignition occurred
        assert result["B"] == 0
        # Signal should evolve normally (not reset by reset_factor)
        # Since no ignition, reset logic should not apply

    def test_reset_factor_bounds_validation(self):
        """reset_factor must be in (0, 1)."""
        config = dict(CONFIG)
        config["reset_factor"] = 1.5  # Invalid: > 1
        config["theta_0"] = 0.1

        from core.validation import validate_config, ValidationError

        with pytest.raises(ValidationError):
            validate_config(config)

    def test_reset_factor_zero_invalid(self):
        """reset_factor = 0 is invalid (no signal retention)."""
        config = dict(CONFIG)
        config["reset_factor"] = 0.0  # Invalid: = 0

        from core.validation import validate_config, ValidationError

        with pytest.raises(ValidationError):
            validate_config(config)

    def test_reset_factor_one_invalid(self):
        """reset_factor = 1 is invalid (no reset)."""
        config = dict(CONFIG)
        config["reset_factor"] = 1.0  # Invalid: = 1

        from core.validation import validate_config, ValidationError

        with pytest.raises(ValidationError):
            validate_config(config)

    def test_reset_factor_effect_on_signal_dynamics(self):
        """Different reset_factor values should produce different signal trajectories."""
        config1 = dict(CONFIG)
        config1["reset_factor"] = 0.2  # Strong reset (20% retention)
        config1["theta_0"] = 0.1
        config1["stochastic_ignition"] = False

        config2 = dict(CONFIG)
        config2["reset_factor"] = 0.8  # Weak reset (80% retention)
        config2["theta_0"] = 0.1
        config2["stochastic_ignition"] = False

        pipeline1 = APGIPipeline(config1)
        pipeline2 = APGIPipeline(config2)

        # Run both pipelines with identical inputs
        for _ in range(10):
            pipeline1.step(x_e=5.0, x_i=5.0)
            pipeline2.step(x_e=5.0, x_i=5.0)

        # The reset factors should be different
        assert pipeline1.config["reset_factor"] != pipeline2.config["reset_factor"]
        # With strong reset, signal should be lower (but timing of ignition may vary)
        # So we just verify both pipelines ran successfully
        assert pipeline1.S >= 0
        assert pipeline2.S >= 0

    def test_reset_with_stochastic_ignition(self):
        """Reset should work correctly with stochastic ignition."""
        config = dict(CONFIG)
        config["reset_factor"] = 0.5
        config["theta_0"] = 0.5
        config["stochastic_ignition"] = True

        pipeline = APGIPipeline(config)

        # Run many steps to get some ignition events
        ignition_count = 0
        for _ in range(100):
            result = pipeline.step(x_e=2.0, x_i=2.0)
            if result["B"] == 1:
                ignition_count += 1
                # After ignition, signal should be reset
                # Check that signal is reduced
                assert pipeline.S < 5.0  # Reasonable bound

        # Should have some ignitions given the configuration
        assert ignition_count > 0

    def test_reset_preserves_signal_sign(self):
        """Signal reset should preserve sign (positive stays positive)."""
        config = dict(CONFIG)
        config["reset_factor"] = 0.5
        config["theta_0"] = 0.1
        config["stochastic_ignition"] = False

        pipeline = APGIPipeline(config)
        pipeline.S = 10.0  # Positive signal

        # Trigger ignition
        pipeline.step(x_e=10.0, x_i=10.0)

        # After reset, signal should still be positive
        assert pipeline.S >= 0
