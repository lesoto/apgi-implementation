import numpy as np
import pytest

from core.ignition import (
    compute_ignition_probability,
    compute_margin,
    detect_ignition_event,
    sample_ignition_state,
)


def test_compute_ignition_probability():
    # S=theta -> prob=0.5
    assert compute_ignition_probability(1.0, 1.0) == 0.5

    # S > theta -> prob > 0.5
    assert compute_ignition_probability(2.0, 1.0) > 0.5

    # S < theta -> prob < 0.5
    assert compute_ignition_probability(0.0, 1.0) < 0.5

    # Large diff (clamping)
    assert compute_ignition_probability(1000.0, 0.0) == 1.0
    assert pytest.approx(compute_ignition_probability(-1000.0, 0.0), abs=1e-10) == 0.0

    with pytest.raises(ValueError, match="tau must be > 0"):
        compute_ignition_probability(1.0, 1.0, tau=0.0)


def test_sample_ignition_state():
    # Deterministic cases
    assert sample_ignition_state(1.0) == 1
    assert sample_ignition_state(0.0) == 0

    # Stochastic case with seed
    rng = np.random.default_rng(42)
    state = sample_ignition_state(0.5, rng=rng)
    assert state in [0, 1]

    with pytest.raises(ValueError, match="p_ignite must be in"):
        sample_ignition_state(1.5)


def test_detect_ignition_event():
    assert detect_ignition_event(1.1, 1.0) is True
    assert detect_ignition_event(0.9, 1.0) is False


def test_compute_margin():
    assert compute_margin(1.5, 1.0) == 0.5
    assert compute_margin(0.5, 1.0) == -0.5
