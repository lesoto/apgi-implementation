"""Tests for main.py CLI entry point and utility functions."""

import json
import os
import tempfile
from unittest.mock import patch

import numpy as np

from main import (
    analyze_signal_statistics,
    generate_synthetic_input,
    main,
    run_multiscale_pipeline,
    run_standard_pipeline,
    save_results,
)


class TestGenerateSyntheticInput:
    """Test synthetic input generation."""

    def test_generate_synthetic_input_returns_tuple(self) -> None:
        """Test that synthetic input returns 4-tuple of floats."""
        result = generate_synthetic_input(t=0)
        assert isinstance(result, tuple)
        assert len(result) == 4
        assert all(isinstance(x, float) for x in result)

    def test_generate_synthetic_input_default_noise(self) -> None:
        """Test synthetic input with default noise."""
        np.random.seed(42)
        x_e, x_hat_e, x_i, x_hat_i = generate_synthetic_input(t=0, noise_std=0.1)
        # At t=0, sin(0) = 0, but noise is added
        assert abs(x_e - x_hat_e) < 0.5  # Allow for noise
        assert abs(x_i - (0.5 + 0.3 * np.cos(0))) < 0.5  # Allow for noise
        assert x_hat_i == 0.5

    def test_generate_synthetic_input_custom_noise(self) -> None:
        """Test synthetic input with custom noise std."""
        np.random.seed(42)
        result1 = generate_synthetic_input(t=10, noise_std=0.01)
        np.random.seed(42)
        result2 = generate_synthetic_input(t=10, noise_std=0.1)
        # Results should differ due to different noise std
        assert result1 != result2

    def test_generate_synthetic_input_periodic_components(self) -> None:
        """Test that synthetic input has periodic components."""
        # Test at different time points to see periodic behavior
        result_t0 = generate_synthetic_input(t=0)
        result_t100 = generate_synthetic_input(t=100)
        # Should be different due to periodic components
        assert result_t0 != result_t100


class TestRunStandardPipeline:
    """Test standard single-scale pipeline execution."""

    def test_run_standard_pipeline_basic(self) -> None:
        """Test basic pipeline execution."""
        results = run_standard_pipeline(n_steps=10, progress_interval=100)
        assert "config" in results
        assert "n_steps" in results
        assert results["n_steps"] == 10
        assert "ignition_count" in results
        assert "ignition_rate" in results
        assert "history" in results
        assert "final_state" in results

    def test_run_standard_pipeline_history_keys(self) -> None:
        """Test that history contains all expected keys."""
        results = run_standard_pipeline(n_steps=5)
        history = results["history"]
        expected_keys = ["S", "theta", "B", "z_e", "z_i", "p_ignite", "C", "V"]
        for key in expected_keys:
            assert key in history
            assert len(history[key]) == 5

    def test_run_standard_pipeline_custom_config(self) -> None:
        """Test pipeline with custom configuration."""
        from config import CONFIG

        custom_config = CONFIG.copy()
        custom_config["lam"] = 0.3
        custom_config["eta"] = 0.05
        results = run_standard_pipeline(n_steps=5, config=custom_config)
        assert results["config"]["lam"] == 0.3
        assert results["config"]["eta"] == 0.05

    def test_run_standard_pipeline_ignition_count(self) -> None:
        """Test that ignition count is tracked."""
        results = run_standard_pipeline(n_steps=100)
        assert "ignition_count" in results
        assert isinstance(results["ignition_count"], int)
        assert 0 <= results["ignition_count"] <= 100

    def test_run_standard_pipeline_final_state(self) -> None:
        """Test final state contains expected fields."""
        results = run_standard_pipeline(n_steps=10)
        final_state = results["final_state"]
        assert "S" in final_state
        assert "theta" in final_state
        assert "sigma2_e" in final_state
        assert "sigma2_i" in final_state


class TestRunMultiscalePipeline:
    """Test multi-scale hierarchical pipeline execution."""

    def test_run_multiscale_pipeline_basic(self) -> None:
        """Test basic multi-scale pipeline."""
        results = run_multiscale_pipeline(n_steps=10, n_levels=3)
        assert "config" in results
        assert "n_steps" in results
        assert "n_levels" in results
        assert results["n_levels"] == 3
        assert "timescales" in results
        assert "weights" in results
        assert "ignition_count" in results
        assert "history" in results

    def test_run_multiscale_pipeline_timescales(self) -> None:
        """Test that timescales are generated correctly."""
        results = run_multiscale_pipeline(n_steps=5, n_levels=3, timescale_k=1.6)
        timescales = results["timescales"]
        assert len(timescales) == 3
        assert timescales[0] == 1.0  # tau0
        # Check geometric progression
        assert timescales[1] == timescales[0] * 1.6
        assert timescales[2] == timescales[1] * 1.6

    def test_run_multiscale_pipeline_weights(self) -> None:
        """Test that weights are normalized."""
        results = run_multiscale_pipeline(n_steps=5, n_levels=3)
        weights = results["weights"]
        assert len(weights) == 3
        # Weights should sum to approximately 1
        assert abs(sum(weights) - 1.0) < 0.01

    def test_run_multiscale_pipeline_history(self) -> None:
        """Test multi-scale history contains expected keys."""
        results = run_multiscale_pipeline(n_steps=5, n_levels=3)
        history = results["history"]
        assert "S_multiscale" in history
        assert "S_standard" in history
        assert "B" in history
        assert "theta" in history
        assert len(history["S_multiscale"]) == 5
        assert len(history["S_standard"]) == 5

    def test_run_multiscale_pipeline_custom_config(self) -> None:
        """Test multi-scale with custom config."""
        from config import CONFIG

        custom_config = CONFIG.copy()
        custom_config["lam"] = 0.3
        results = run_multiscale_pipeline(n_steps=5, config=custom_config)
        assert results["config"]["lam"] == 0.3


class TestAnalyzeSignalStatistics:
    """Test signal statistics analysis."""

    def test_analyze_signal_statistics_basic(self) -> None:
        """Test basic statistics computation."""
        signal = [1.0, 2.0, 3.0, 4.0, 5.0]
        stats = analyze_signal_statistics(signal, label="Test")
        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert "range" in stats
        assert stats["mean"] == 3.0
        assert stats["min"] == 1.0
        assert stats["max"] == 5.0
        assert stats["range"] == 4.0

    def test_analyze_signal_statistics_hurst_insufficient_data(self) -> None:
        """Test Hurst exponent with insufficient data."""
        signal = [1.0, 2.0, 3.0]
        stats = analyze_signal_statistics(signal, label="Test")
        assert "hurst_exponent" not in stats  # Should not be computed
        assert "mean" in stats

    def test_analyze_signal_statistics_hurst_sufficient_data(self) -> None:
        """Test Hurst exponent with sufficient data."""
        # Generate 256 samples of random walk
        np.random.seed(42)
        signal = np.cumsum(np.random.randn(300)).tolist()
        stats = analyze_signal_statistics(signal, label="Test")
        # Hurst should be computed
        assert "hurst_exponent" in stats
        assert isinstance(stats["hurst_exponent"], float)

    def test_analyze_signal_statistics_empty_signal(self) -> None:
        """Test statistics with empty signal (handles gracefully)."""
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            try:
                stats = analyze_signal_statistics([], label="Test")
                # If it doesn't raise, check structure
                assert "mean" in stats  # pragma: no cover
            except (ValueError, RuntimeWarning):
                # Expected for empty array
                pass

    def test_analyze_signal_statistics_runtime_warning(self) -> None:
        """Test statistics with RuntimeWarning (handles gracefully)."""
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            try:
                stats = analyze_signal_statistics([1.0], label="Test")
                # If it doesn't raise, check structure
                assert "mean" in stats  # pragma: no cover
            except (ValueError, RuntimeWarning):  # pragma: no cover
                # Expected for problematic data
                pass

    def test_analyze_signal_statistics_single_value(self) -> None:
        """Test statistics with single value."""
        stats = analyze_signal_statistics([1.0], label="Test")
        assert "mean" in stats
        assert stats["mean"] == 1.0


class TestSaveResults:
    """Test results saving functionality."""

    def test_save_results_creates_file(self) -> None:
        """Test that save_results creates a file."""
        results = {"test": "data", "value": 42}
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            filepath = f.name

        try:
            save_results(results, filepath)
            assert os.path.exists(filepath)
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)

    def test_save_results_valid_json(self) -> None:
        """Test that saved file is valid JSON."""
        results = {
            "config": {"lam": 0.2},
            "n_steps": 100,
            "history": {"S": [1.0, 2.0, 3.0]},
        }
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            filepath = f.name

        try:
            save_results(results, filepath)
            with open(filepath, "r") as f:
                loaded = json.load(f)
            assert loaded["config"]["lam"] == 0.2
            assert loaded["n_steps"] == 100
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)

    def test_save_results_numpy_conversion(self) -> None:
        """Test that numpy arrays are converted to lists."""
        results = {
            "array": np.array([1.0, 2.0, 3.0]),
            "nested": {"inner_array": np.array([4.0, 5.0])},
        }
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            filepath = f.name

        try:
            save_results(results, filepath)
            with open(filepath, "r") as f:
                loaded = json.load(f)
            assert isinstance(loaded["array"], list)
            assert loaded["array"] == [1.0, 2.0, 3.0]
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)


class TestMainFunction:
    """Test main CLI entry point."""

    @patch("sys.argv", ["main.py", "--steps", "10"])
    def test_main_default_mode(self) -> None:
        """Test main with default arguments."""
        result = main()
        assert result == 0

    @patch("sys.argv", ["main.py", "--steps", "5", "--seed", "42"])
    def test_main_with_seed(self) -> None:
        """Test main with random seed."""
        result = main()
        assert result == 0

    @patch("sys.argv", ["main.py", "--steps", "5", "--beta", "0.5"])
    def test_main_with_beta(self) -> None:
        """Test main with dopamine bias."""
        result = main()
        assert result == 0

    @patch("sys.argv", ["main.py", "--steps", "5", "--stochastic"])
    def test_main_stochastic_ignition(self) -> None:
        """Test main with stochastic ignition."""
        result = main()
        assert result == 0

    @patch("sys.argv", ["main.py", "--steps", "5", "--ne-on-threshold"])
    def test_main_ne_on_threshold(self) -> None:
        """Test main with NE on threshold."""
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = main()
            assert result == 0

    @patch("sys.argv", ["main.py", "--multiscale", "--steps", "5", "--levels", "3"])
    def test_main_multiscale_mode(self) -> None:
        """Test main in multiscale mode."""
        result = main()
        assert result == 0

    @patch(
        "sys.argv",
        ["main.py", "--multiscale", "--steps", "5", "--levels", "3", "--k", "2.0"],
    )
    def test_main_multiscale_custom_k(self) -> None:
        """Test main with custom timescale factor."""
        result = main()
        assert result == 0

    @patch("sys.argv", ["main.py", "--steps", "5", "--demo"])
    def test_main_demo_flag(self) -> None:
        """Test main with demo flag."""
        result = main()
        assert result == 0

    def test_main_keyboard_interrupt(self) -> None:
        """Test main handles KeyboardInterrupt."""
        with patch("sys.argv", ["main.py", "--steps", "5"]):
            with patch("main.run_standard_pipeline", side_effect=KeyboardInterrupt):
                result = main()
                assert result == 130

    def test_main_exception_handling(self) -> None:
        """Test main handles exceptions gracefully."""
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            with patch("sys.argv", ["main.py", "--steps", "-1"]):  # Invalid negative steps
                result = main()
                assert result == 1
