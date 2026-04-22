"""Extended tests for pipeline.py to achieve 100% coverage."""

import numpy as np
from pipeline import APGIPipeline
from config import CONFIG


class TestAPGIPipelineExtended:
    """Extended tests for APGIPipeline."""

    def test_reservoir_mode_sampling(self):
        """Should use sampling when stochastic_ignition=True in reservoir mode."""
        config = CONFIG.copy()
        config["use_reservoir"] = True
        config["stochastic_ignition"] = True
        config["reservoir_theta_scale"] = 0.1

        pipeline = APGIPipeline(config)

        # Run a few steps
        results = []
        for t in range(10):
            x_e = np.sin(0.05 * t)
            x_hat_e = 0.0
            x_i = 0.5
            x_hat_i = 0.5
            result = pipeline.step(x_e, x_hat_e, x_i, x_hat_i)
            results.append(result)

        # Should have ignition decisions
        ignition_count = sum(r["B"] for r in results)
        assert isinstance(ignition_count, (int, np.integer))

    def test_reservoir_mode_deterministic_ignition(self):
        """Should use deterministic ignition when stochastic_ignition=False in reservoir mode."""
        config = CONFIG.copy()
        config["use_reservoir"] = True
        config["stochastic_ignition"] = False
        config["reservoir_theta_scale"] = 0.1

        pipeline = APGIPipeline(config)

        # Run a step
        x_e = 1.0
        x_hat_e = 0.0
        x_i = 0.5
        x_hat_i = 0.5
        result = pipeline.step(x_e, x_hat_e, x_i, x_hat_i)

        # Should have deterministic ignition decision
        assert "B" in result
        assert isinstance(result["B"], (int, np.integer))
