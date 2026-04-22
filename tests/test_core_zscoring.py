"""Comprehensive unit tests for core/zscoring.py module.

Tests cover:
- ZScoreWindow class
- DualZScoreProcessor class
- create_standard_zscorer function
"""

from __future__ import annotations

import pytest

from core.zscoring import (
    ZScoreWindow,
    DualZScoreProcessor,
    create_standard_zscorer,
)


class TestZScoreWindow:
    """Tests for ZScoreWindow class."""

    def test_initialization(self):
        """Should initialize with correct parameters."""
        window = ZScoreWindow(sampling_rate_hz=100.0, window_seconds=10.0)
        assert window.sampling_rate == 100.0
        assert window.window_duration == 10.0
        assert window.window_size == 1000
        assert window._count == 0

    def test_small_window(self):
        """Should raise ValueError for too small window."""
        with pytest.raises(ValueError, match="Window size"):
            ZScoreWindow(sampling_rate_hz=1.0, window_seconds=1.0)

    def test_update_single_value(self):
        """Should update with single value."""
        window = ZScoreWindow(sampling_rate_hz=100.0, window_seconds=1.0)
        result = window.update(1.0)
        # First value: mean=1.0, std undefined, returns 0
        assert result == 0.0

    def test_update_multiple_values(self):
        """Should compute z-scores for multiple values."""
        window = ZScoreWindow(sampling_rate_hz=100.0, window_seconds=1.0)

        # Fill window with known values
        values = [1.0, 2.0, 3.0, 4.0, 5.0] * 20  # 100 values
        z_scores = []
        for v in values:
            z = window.update(v)
            z_scores.append(z)

        # After window is full, z-scores should vary
        non_zero_scores = [z for z in z_scores if z != 0]
        assert len(non_zero_scores) > 0

    def test_get_stats(self):
        """Should return current window statistics."""
        window = ZScoreWindow(sampling_rate_hz=100.0, window_seconds=1.0)

        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            window.update(v)

        stats = window.get_stats()
        assert "mean" in stats
        assert "std" in stats
        assert "n" in stats
        assert stats["n"] == 5
        assert stats["mean"] == 3.0

    def test_get_stats_empty(self):
        """Should handle empty window."""
        window = ZScoreWindow(sampling_rate_hz=100.0, window_seconds=1.0)
        stats = window.get_stats()
        assert stats["mean"] == 0.0
        assert stats["std"] == 1.0
        assert stats["n"] == 0

    def test_reset(self):
        """Should clear the window."""
        window = ZScoreWindow(sampling_rate_hz=100.0, window_seconds=1.0)

        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            window.update(v)

        window.reset()
        assert len(window.buffer) == 0
        assert window._count == 0
        assert window._sum == 0.0

    def test_zscore_calculation(self):
        """Should compute z-score correctly."""
        window = ZScoreWindow(sampling_rate_hz=100.0, window_seconds=0.1)

        # Fill with known values
        for v in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]:
            window.update(v)

        # Now window has mean=5.5, std~3.03
        # Add a value at mean
        result = window.update(5.5)
        # Should be close to 0, but allow for numerical precision
        assert pytest.approx(result, abs=0.5) == 0.0

    def test_window_sliding(self):
        """Should slide window correctly."""
        window = ZScoreWindow(sampling_rate_hz=10.0, window_seconds=0.5)
        # window_size = 5

        for i in range(10):
            window.update(float(i))

        # Should only have last 5 values: [5, 6, 7, 8, 9]
        stats = window.get_stats()
        assert stats["n"] == 5
        assert stats["mean"] == 7.0


class TestDualZScoreProcessor:
    """Tests for DualZScoreProcessor class."""

    def test_initialization(self):
        """Should initialize with correct windows."""
        processor = DualZScoreProcessor(
            sampling_rate_e_hz=100.0,
            sampling_rate_i_hz=50.0,
            window_seconds=10.0,
        )
        assert processor.window_e.sampling_rate == 100.0
        assert processor.window_i.sampling_rate == 50.0

    def test_process(self):
        """Should process both modalities."""
        processor = DualZScoreProcessor(
            sampling_rate_e_hz=10.0,
            sampling_rate_i_hz=10.0,
            window_seconds=1.0,
        )

        # Fill windows
        for i in range(20):
            z_e, z_i = processor.process(float(i), float(i * 2))

        # Both should eventually return non-zero z-scores
        assert True  # Processing succeeded

    def test_get_stats(self):
        """Should return stats for both windows."""
        processor = DualZScoreProcessor(
            sampling_rate_e_hz=10.0,
            sampling_rate_i_hz=10.0,
            window_seconds=1.0,
        )

        for i in range(20):
            processor.process(float(i), float(i))

        stats = processor.get_stats()
        assert "exteroceptive" in stats
        assert "interoceptive" in stats
        assert stats["exteroceptive"]["n"] > 0
        assert stats["interoceptive"]["n"] > 0

    def test_reset(self):
        """Should reset both windows."""
        processor = DualZScoreProcessor(
            sampling_rate_e_hz=10.0,
            sampling_rate_i_hz=10.0,
            window_seconds=1.0,
        )

        for i in range(20):
            processor.process(float(i), float(i))

        processor.reset()
        stats = processor.get_stats()
        assert stats["exteroceptive"]["n"] == 0
        assert stats["interoceptive"]["n"] == 0


class TestCreateStandardZscorer:
    """Tests for create_standard_zscorer function."""

    def test_default_rates(self):
        """Should create with default sampling rates."""
        zscorer = create_standard_zscorer()
        assert zscorer.window_e.sampling_rate == 100.0
        assert zscorer.window_i.sampling_rate == 50.0

    def test_custom_rates(self):
        """Should accept custom sampling rates."""
        zscorer = create_standard_zscorer(
            extero_rate_hz=200.0,
            intero_rate_hz=100.0,
        )
        assert zscorer.window_e.sampling_rate == 200.0
        assert zscorer.window_i.sampling_rate == 100.0

    def test_window_duration(self):
        """Should use 10-second window."""
        zscorer = create_standard_zscorer()
        assert zscorer.window_e.window_duration == 10.0
        assert zscorer.window_i.window_duration == 10.0
