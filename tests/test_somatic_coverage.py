import pytest
import numpy as np
from core.somatic import (
    somatic_marker_arousal,
    somatic_marker_valence,
    compute_precision_with_somatic_marker,
    compute_somatic_gain,
    update_somatic_marker_euler,
)


def test_somatic_marker_arousal():
    assert somatic_marker_arousal(0.5) == 0.0
    assert somatic_marker_arousal(1.0) == 2.0
    assert somatic_marker_arousal(0.0) == -2.0
    assert somatic_marker_arousal(2.0) == 2.0  # clamped


def test_somatic_marker_valence():
    # Circumplex: M = 2 * valence * (0.5 + arousal)
    # M = 2 * 1.0 * (0.5 + 0.5) = 2.0
    assert somatic_marker_valence(1.0, 0.5) == 2.0
    # M = 2 * -1.0 * (0.5 + 0.0) = -1.0
    assert somatic_marker_valence(-1.0, 0.0) == -1.0
    # Clamping
    assert somatic_marker_valence(1.0, 1.0) == 2.0  # 2 * 1 * 1.5 = 3.0 -> 2.0


def test_compute_precision_with_somatic_marker():
    # pi = 10 * exp(0.5 * 2) = 10 * e ~ 27.18
    res = compute_precision_with_somatic_marker(10.0, 0.5, 2.0)
    assert pytest.approx(res) == 10.0 * np.exp(1.0)
    # Clamping
    assert compute_precision_with_somatic_marker(1.0, 10.0, 2.0, pi_max=100.0) == 100.0


def test_compute_somatic_gain():
    assert pytest.approx(compute_somatic_gain(2.0, beta=0.5)) == np.exp(1.0)


def test_update_somatic_marker_euler():
    # M=0.0, target_arousal=1.0 -> M_target=2.0
    # dM = -(0 - 2.0) / 100.0 * 10.0 = 2.0 / 10 = 0.2
    # M_new = 0.0 + 0.2 = 0.2
    res = update_somatic_marker_euler(0.0, 1.0, tau_M=100.0, dt=10.0)
    assert pytest.approx(res) == 0.2
