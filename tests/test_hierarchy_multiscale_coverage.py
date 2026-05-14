import pytest
import numpy as np
from hierarchy.multiscale import (
    build_timescales,
    estimate_optimal_timescale_ratio,
    estimate_hierarchy_levels_from_data,
    update_multiscale_feature,
    multiscale_weights,
    aggregate_multiscale_signal,
    aggregate_multiscale_signal_phi,
    apply_reset_rule,
    phase_signal,
    modulate_threshold,
    bottom_up_cascade,
)


def test_build_timescales():
    ts = build_timescales(1.0, 2.0, 3)
    assert np.array_equal(ts, [1.0, 2.0, 4.0])
    with pytest.raises(ValueError, match="tau0"):
        build_timescales(0, 2.0, 3)
    with pytest.raises(ValueError, match="k"):
        build_timescales(1.0, 1.0, 3)
    with pytest.raises(ValueError, match="n_levels"):
        build_timescales(1.0, 2.0, 0)


def test_estimate_optimal_timescale_ratio():
    # Signal with characteristic scales
    t = np.linspace(0, 10, 1000)
    sig = np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 10 * t)
    k = estimate_optimal_timescale_ratio(sig, fs=100.0)
    assert 1.3 <= k <= 2.0

    # Not enough peaks (using short zero signal)
    k_def = estimate_optimal_timescale_ratio(np.zeros(5))
    assert k_def == 1.6


def test_estimate_hierarchy_levels_from_data():
    sig = np.random.randn(200)
    n = estimate_hierarchy_levels_from_data(sig)
    assert 2 <= n <= 8


def test_update_multiscale_feature():
    val = update_multiscale_feature(1.0, 2.0, 10.0)
    assert isinstance(val, float)
    with pytest.raises(ValueError, match="tau_i"):
        update_multiscale_feature(1.0, 2.0, 0)


def test_multiscale_weights():
    w = multiscale_weights(3, 2.0)
    assert len(w) == 3
    assert pytest.approx(np.sum(w)) == 1.0


def test_aggregate_multiscale_signal():
    phi = [1.0, 2.0]
    pi = [1.0, 1.0]
    w = [0.6, 0.4]
    s = aggregate_multiscale_signal(phi, pi, w)
    assert s == pytest.approx(0.6 * 1.0 * 1.0 + 0.4 * 1.0 * 2.0)

    with pytest.raises(ValueError, match="same length"):
        aggregate_multiscale_signal([1], [1, 2], [1])


def test_aggregate_multiscale_signal_phi():
    phi = [-1.0, 2.0]
    pi = [1.0, 1.0]
    w = [0.5, 0.5]
    s = aggregate_multiscale_signal_phi(phi, pi, w)
    assert s == pytest.approx(0.5 * (-1.0) + 0.5 * 2.0)

    with pytest.raises(ValueError, match="same length"):
        aggregate_multiscale_signal_phi([1], [1, 2], [1])


def test_apply_reset_rule():
    s, th = apply_reset_rule(1.0, 1.0, rho=0.1, delta=2.0)
    assert s == 0.1
    assert th == 3.0


def test_phase_signal():
    p = phase_signal(np.pi, 1.0, phi0=0.0)
    assert p == pytest.approx(np.pi)


def test_modulate_threshold():
    th = modulate_threshold(1.0, 1.0, 0.0, 0.1)
    assert th == pytest.approx(1.1)


def test_bottom_up_cascade():
    th = bottom_up_cascade(1.0, 1.5, 1.0, 0.1)
    assert th == pytest.approx(0.9)
    th_low = bottom_up_cascade(1.0, 0.5, 1.0, 0.1)
    assert th_low == 1.0
