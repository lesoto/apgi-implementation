from __future__ import annotations

import numpy as np

from stats.avalanche import (
    extract_avalanches,
    fit_discrete_power_law_mle,
    validate_avalanche_power_law,
)


def test_extract_avalanches_empty() -> None:
    sizes = extract_avalanches(np.zeros(100))
    assert sizes.size == 0


def test_extract_avalanches_basic_runs() -> None:
    activity = np.array([0, 1, 1, 0, 1, 0, 1, 1, 1, 0])
    sizes = extract_avalanches(activity)
    assert sizes.tolist() == [2, 1, 3]


def test_fit_discrete_power_law_mle_insufficient() -> None:
    fit = fit_discrete_power_law_mle(np.array([]), xmin=1)
    assert np.isnan(fit.alpha)
    assert fit.n == 0


def test_validate_avalanche_power_law_insufficient_data() -> None:
    activity = np.array([0, 1, 0, 1, 0, 1, 0])  # 3 avalanches
    result = validate_avalanche_power_law(activity, min_avalanches=10)
    assert result["status"] == "insufficient_data"
    assert result["n_avalanches"] == 3
