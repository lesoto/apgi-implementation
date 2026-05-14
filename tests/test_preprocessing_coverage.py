import numpy as np
import pytest

from core.preprocessing import (
    EMAStats,
    RunningStats,
    compute_prediction_error,
    normalize_error,
    update_prediction,
    z_score,
)


def test_running_stats():
    rs = RunningStats(window_size=3)
    assert rs.mean() == 0.0
    assert rs.variance() == 1.0

    rs.update(1.0)
    assert rs.mean() == 1.0
    # For single element with Bessel correction, var is NaN/Undefined in some implementations,
    # but np.var(..., ddof=1) of single element returns NaN.
    # Our implementation returns what np.var returns.

    rs.update(2.0)
    # mean = 1.5, var = ((1-1.5)**2 + (2-1.5)**2) / 1 = 0.5
    assert rs.mean() == 1.5
    assert pytest.approx(rs.variance()) == 0.5
    assert pytest.approx(rs.std()) == np.sqrt(0.5)

    # MLE variance (ddof=0)
    # var = (0.25 + 0.25) / 2 = 0.25
    assert pytest.approx(rs.variance(bessel_correction=False)) == 0.25

    with pytest.raises(ValueError, match="window_size must be > 0"):
        RunningStats(0)


def test_ema_stats():
    ema = EMAStats(alpha=0.1, initial_mean=0.0, initial_var=1.0)
    # update with 1.0
    # mean = 0.9*0 + 0.1*1 = 0.1
    # var = 0.9*1 + 0.1*(1 - 0.1)**2 = 0.9 + 0.1*0.81 = 0.9 + 0.081 = 0.981
    ema.update(1.0)
    assert pytest.approx(ema.mean()) == 0.1
    assert pytest.approx(ema.variance()) == 0.981
    assert pytest.approx(ema.std()) == np.sqrt(0.981)

    with pytest.raises(ValueError, match="alpha must be in"):
        EMAStats(1.5)


def test_prediction_helpers():
    assert compute_prediction_error(1.0, 0.8) == pytest.approx(0.2)
    # x_hat = 0.8 + 0.1 * 10 * 0.2 = 0.8 + 0.2 = 1.0
    assert update_prediction(0.8, 0.2, 10.0, 0.1) == pytest.approx(1.0)


def test_normalization():
    # 1.0 / (0.5 + 1e-8)
    assert pytest.approx(normalize_error(1.0, 0.5)) == 1.0 / (0.5 + 1e-8)

    rs = RunningStats(window_size=10)
    rs.update(1.0)
    rs.update(3.0)  # mean=2.0, std=sqrt(2)
    # (2.0 - 2.0) / std = 0
    assert z_score(2.0, rs) == 0.0
