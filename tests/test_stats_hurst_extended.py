"""Extended tests for stats/hurst.py to achieve 100% coverage."""

import numpy as np
import pytest
from stats.hurst import (
    estimate_beta_welch,
)


class TestEstimateBetaWelchExtended:
    """Extended tests for estimate_beta_welch."""

    def test_insufficient_points_error(self):
        """Should raise ValueError when insufficient frequency points."""
        # Create a signal that results in very few frequency points after filtering
        signal = np.zeros(10)  # Very short signal
        fs = 100.0

        # This should trigger the error path at line 69
        with pytest.raises(ValueError, match="need at least 2 frequency points"):
            estimate_beta_welch(signal, fs, fmin=40.0, fmax=50.0)
