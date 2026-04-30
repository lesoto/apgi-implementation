"""Test suite for Landauer scaling consistency (§11).

Verifies the $10^{20}$ scaling factor used to map physical Joules
to dimensionless allostatic units.
"""

import pytest

from core.thermodynamics import K_BOLTZMANN, T_ENV_DEFAULT, compute_landauer_cost
from core.threshold import compute_metabolic_cost_realistic


def test_landauer_scaling_magnitude():
    """Verify that scaled Landauer cost is in the order of dimensionless signal costs."""
    S = 1.0
    eps = 1e-6
    kappa_meta = 1.0

    # 1) Physical energy in Joules
    e_phys = compute_landauer_cost(S, eps, K_BOLTZMANN, T_ENV_DEFAULT, kappa_meta)

    # Expected bits: log2(1.0/1e-6) approx 19.9
    # E_min = N * k_B * T * ln(2)
    # E_min approx 19.9 * 1.38e-23 * 310 * 0.693 approx 5.9e-20 J

    assert 1e-21 < e_phys < 1e-18

    # 2) Scaled energy in allostatic units
    scale_factor = 1e20
    e_scaled = e_phys * scale_factor

    # Should be in order of 1-100 dimensionless units
    assert 0.1 < e_scaled < 100.0


def test_metabolic_cost_landauer_enforcement():
    """Test that compute_metabolic_cost_realistic correctly applies the scaling."""
    S = 5.0
    B_prev = 0
    c1 = 0.1
    c2 = 1.0

    # Base cost = c1*S = 0.5
    cost_no_landauer = compute_metabolic_cost_realistic(S, B_prev, c1, c2, enforce_landauer=False)
    assert cost_no_landauer == pytest.approx(0.5)

    # With landauer enforcement
    cost_with_landauer = compute_metabolic_cost_realistic(
        S, B_prev, c1, c2, enforce_landauer=True, kappa_meta=1.0
    )

    # Compute manually:
    # N = log2(5.0 / 1e-6) = 22.25
    # E = 22.25 * 1.38e-23 * 310 * 0.693 = 6.6e-20
    # Scaled E = 6.6
    # max(0.5, 6.6) = 6.6

    assert cost_with_landauer > cost_no_landauer
    assert 5.0 < cost_with_landauer < 10.0


def test_scaling_consistency_across_S():
    """Verify that scaling maintains the logarithmic relationship."""
    S_vals = [0.1, 1.0, 10.0, 100.0]
    eps = 1e-6
    scale_factor = 1e20

    costs = []
    for s in S_vals:
        e = compute_landauer_cost(s, eps)
        costs.append(e * scale_factor)

    # Differences should correspond to bits added (log2 step of 10 is ~3.32 bits)
    # Each 10x increase in S adds ~3.3 bits.
    # Energy per bit: k_B * T * ln(2) * 1e20 approx 1.38e-23 * 310 * 0.693 * 1e20 approx 0.3
    # So each 10x step should add approx 3.3 * 0.3 = 1.0 units

    for i in range(len(costs) - 1):
        diff = costs[i + 1] - costs[i]
        assert 0.5 < diff < 1.5
