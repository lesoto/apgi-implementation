import pytest
from core.zscoring import ZScoreWindow, DualZScoreProcessor, create_standard_zscorer


def test_zscore_window():
    # Window size = 2
    zw = ZScoreWindow(sampling_rate_hz=1.0, window_seconds=2.0)
    assert zw.window_size == 2

    # 1st sample: mean=1.0, var=0.0, std=0.0 -> returns 0.0
    assert zw.update(1.0) == 0.0

    # 2nd sample: [1.0, 3.0] -> mean=2.0, var=(1+9)/2 - 4 = 1.0, std=1.0
    # (3.0 - 2.0) / (1.0 + eps) ~ 1.0
    assert pytest.approx(zw.update(3.0)) == 1.0

    # 3rd sample (overflow): [3.0, 5.0] -> mean=4.0, var=(9+25)/2 - 16 = 1.0, std=1.0
    # (5.0 - 4.0) / (1.0 + eps) ~ 1.0
    assert pytest.approx(zw.update(5.0)) == 1.0

    # Reset
    zw.reset()
    assert zw._count == 0
    assert zw.get_stats()["n"] == 0
    assert zw.get_stats()["std"] == 1.0


def test_zscore_window_small():
    with pytest.raises(ValueError, match="Window size 1 too small"):
        ZScoreWindow(sampling_rate_hz=1.0, window_seconds=1.0)


def test_dual_zscore_processor():
    dz = DualZScoreProcessor(sampling_rate_e_hz=1.0, sampling_rate_i_hz=1.0, window_seconds=2.0)
    z_e, z_i = dz.process(1.0, 10.0)
    assert z_e == 0.0
    assert z_i == 0.0

    stats = dz.get_stats()
    assert stats["exteroceptive"]["mean"] == 1.0
    assert stats["interoceptive"]["mean"] == 10.0

    dz.reset()
    assert dz.window_e._count == 0


def test_create_standard_zscorer():
    dz = create_standard_zscorer()
    assert dz.window_e.window_size == 1000
    assert dz.window_i.window_size == 500
