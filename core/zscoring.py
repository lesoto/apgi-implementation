"""Z-scoring with 10-second window for preprocessing.

Implements the spec requirement for within-modality z-scoring:
z = (ϵ - μ) / σ where μ, σ estimated from running 10-second window.
"""

from __future__ import annotations

from collections import deque

import numpy as np


class ZScoreWindow:
    """Running z-score calculator with configurable window duration.

    Implements: z = (ϵ - μ) / σ
    where μ, σ are estimated from a running window of recent samples.
    """

    def __init__(
        self,
        sampling_rate_hz: float = 100.0,
        window_seconds: float = 10.0,
        eps: float = 1e-8,
    ):
        """Initialize z-score window.

        Args:
            sampling_rate_hz: Sampling frequency in Hz (e.g., 100 for 100Hz)
            window_seconds: Window duration in seconds (default: 10s)
            eps: Numerical stability epsilon
        """

        self.sampling_rate = sampling_rate_hz
        self.window_duration = window_seconds
        self.eps = eps

        # Calculate window size in samples
        self.window_size = int(sampling_rate_hz * window_seconds)
        if self.window_size < 2:
            raise ValueError(
                f"Window size {self.window_size} too small. "
                f"Need sampling_rate * window_seconds >= 2"
            )

        # Circular buffer for efficient windowed statistics
        self.buffer: deque[float] = deque(maxlen=self.window_size)
        self._sum = 0.0
        self._sum_sq = 0.0
        self._count = 0

    def update(self, epsilon: float) -> float:
        """Add new sample and return z-scored value.

        Args:
            epsilon: Raw prediction error (ϵ)

        Returns:
            Z-scored value z = (ϵ - μ) / σ
        """

        # If window full, remove oldest
        if len(self.buffer) >= self.window_size:
            old = self.buffer[0]
            self._sum -= old
            self._sum_sq -= old * old
            self._count -= 1

        # Add new value
        self.buffer.append(float(epsilon))
        self._sum += float(epsilon)
        self._sum_sq += float(epsilon) * float(epsilon)
        self._count += 1

        # Compute statistics
        mean = self._sum / self._count if self._count > 0 else 0.0
        variance = (self._sum_sq / self._count - mean**2) if self._count > 0 else 1.0
        std = np.sqrt(max(variance, 0.0))

        # Return z-score
        if std < self.eps:
            return 0.0
        return float((epsilon - mean) / (std + self.eps))

    def get_stats(self) -> dict:
        """Get current window statistics."""

        if self._count == 0:
            return {"mean": 0.0, "std": 1.0, "n": 0}

        mean = self._sum / self._count
        variance = self._sum_sq / self._count - mean**2
        std = np.sqrt(max(variance, 0.0))

        return {"mean": float(mean), "std": float(std), "n": self._count}

    def reset(self):
        """Clear the window."""

        self.buffer.clear()
        self._sum = 0.0
        self._sum_sq = 0.0
        self._count = 0


class DualZScoreProcessor:
    """Dual-channel z-score processor for exteroceptive and interoceptive signals.

    Maintains separate 10-second windows for each modality with potentially
    different sampling rates.
    """

    def __init__(
        self,
        sampling_rate_e_hz: float = 100.0,
        sampling_rate_i_hz: float = 50.0,
        window_seconds: float = 10.0,
    ):
        """Initialize dual-channel processor.

        Args:
            sampling_rate_e_hz: Exteroceptive sampling rate (Hz)
            sampling_rate_i_hz: Interoceptive sampling rate (Hz)
            window_seconds: Window duration (default: 10s)
        """

        self.window_e = ZScoreWindow(
            sampling_rate_hz=sampling_rate_e_hz,
            window_seconds=window_seconds,
        )
        self.window_i = ZScoreWindow(
            sampling_rate_hz=sampling_rate_i_hz,
            window_seconds=window_seconds,
        )

    def process(
        self,
        epsilon_e: float,
        epsilon_i: float,
    ) -> tuple[float, float]:
        """Process both modalities and return z-scored errors.

        Args:
            epsilon_e: Exteroceptive prediction error
            epsilon_i: Interoceptive prediction error

        Returns:
            (z_e, z_i) - z-scored errors
        """

        z_e = self.window_e.update(epsilon_e)
        z_i = self.window_i.update(epsilon_i)

        return z_e, z_i

    def get_stats(self) -> dict:
        """Get statistics for both windows."""

        return {
            "exteroceptive": self.window_e.get_stats(),
            "interoceptive": self.window_i.get_stats(),
        }

    def reset(self):
        """Reset both windows."""

        self.window_e.reset()
        self.window_i.reset()


def create_standard_zscorer(
    extero_rate_hz: float = 100.0,
    intero_rate_hz: float = 50.0,
) -> DualZScoreProcessor:
    """Create standard 10-second window z-score processor.

    Args:
        extero_rate_hz: Exteroceptive sampling rate (default: 100Hz)
        intero_rate_hz: Interoceptive sampling rate (default: 50Hz)

    Returns:
        Configured DualZScoreProcessor
    """

    return DualZScoreProcessor(
        sampling_rate_e_hz=extero_rate_hz,
        sampling_rate_i_hz=intero_rate_hz,
        window_seconds=10.0,
    )
