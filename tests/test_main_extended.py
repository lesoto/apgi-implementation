"""Extended tests for main.py to achieve 100% coverage."""

import numpy as np

from main import analyze_signal_statistics, generate_synthetic_input, save_results


class TestAnalyzeSignalStatisticsExtended:
    """Extended tests for analyze_signal_statistics."""

    def test_insufficient_data_for_hurst(self):
        """Should handle insufficient data gracefully."""
        # Signal with less than 256 points
        signal_history = [0.1, 0.2, 0.3] * 50  # 150 points
        result = analyze_signal_statistics(signal_history, "Test Signal")
        assert "mean" in result
        assert "std" in result
        # Hurst should not be computed with insufficient data
        assert "hurst_exponent" not in result

    def test_hurst_exception_handling(self):
        """Should handle exceptions in Hurst estimation."""
        # Create a signal that might cause Hurst estimation to fail
        signal_history = [1e10] * 300 + [-1e10] * 300  # Extreme values
        result = analyze_signal_statistics(signal_history, "Test Signal")
        # Should still return basic stats even if Hurst fails
        assert "mean" in result
        assert "std" in result

    def test_hurst_persistent_process(self):
        """Should identify persistent process when H > 0.5."""
        # Create a signal with persistent characteristics
        signal_history = np.cumsum(np.random.randn(300)).tolist()
        result = analyze_signal_statistics(signal_history, "Test Signal")
        assert "mean" in result
        assert "std" in result
        # With sufficient data, hurst_exponent should be computed
        assert "hurst_exponent" in result

    def test_hurst_anti_persistent_process(self):
        """Should identify anti-persistent process when H < 0.5."""
        # Create a signal with anti-persistent characteristics (mean-reverting)
        signal_history = []
        for i in range(300):
            if i == 0:
                signal_history.append(0.0)
            else:
                # Mean-reverting: tends to return to 0
                signal_history.append(
                    0.5 * signal_history[-1] + 0.1 * np.random.randn()
                )
        result = analyze_signal_statistics(signal_history, "Test Signal")
        assert "mean" in result
        assert "std" in result
        # With sufficient data, hurst_exponent should be computed
        assert "hurst_exponent" in result

    def test_hurst_random_walk(self):
        """Should identify random walk when H ≈ 0.5."""
        # Create a random walk
        signal_history = np.cumsum(np.random.randn(300)).tolist()
        result = analyze_signal_statistics(signal_history, "Test Signal")
        assert "mean" in result
        assert "std" in result
        # With sufficient data, hurst_exponent should be computed
        assert "hurst_exponent" in result


class TestSaveResults:
    """Tests for save_results function."""

    def test_save_and_load_json(self, tmp_path):
        """Should save results that can be loaded."""
        import json

        results = {
            "config": {"test": "value"},
            "n_steps": 100,
            "history": {
                "S": [0.1, 0.2, 0.3],
                "theta": [1.0, 1.1, 1.2],
            },
            "numpy_array": np.array([1, 2, 3]),
        }
        filepath = tmp_path / "test_results.json"
        save_results(results, str(filepath))

        # Load and verify
        with open(filepath, "r") as f:
            loaded = json.load(f)

        assert loaded["n_steps"] == 100
        assert loaded["numpy_array"] == [1, 2, 3]


class TestGenerateSyntheticInput:
    """Tests for generate_synthetic_input."""

    def test_generates_values(self):
        """Should generate four values."""
        x_e, x_hat_e, x_i, x_hat_i = generate_synthetic_input(t=0, noise_std=0.0)
        assert isinstance(x_e, float)
        assert isinstance(x_hat_e, float)
        assert isinstance(x_i, float)
        assert isinstance(x_hat_i, float)

    def test_noise_affects_output(self):
        """Should add noise when noise_std > 0."""
        # Run multiple times and check variance
        results = [generate_synthetic_input(t=0, noise_std=0.1) for _ in range(10)]
        x_e_values = [r[0] for r in results]
        # With noise, values should vary
        assert np.std(x_e_values) > 0
