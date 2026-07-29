"""Tests for core.validation canonical parameter range checking (Notation Appendix)."""

import pytest

from core.validation import (
    CANONICAL_PARAMETER_RANGES,
    check_canonical_compliance_for_config,
    check_canonical_parameter_ranges,
)


def test_all_canonical_ranges_well_formed():
    for name, (lo, hi) in CANONICAL_PARAMETER_RANGES.items():
        assert lo < hi, f"{name}: lo must be < hi"


def test_check_in_range():
    r = check_canonical_parameter_ranges(gamma_sig=5.0, theta_0=0.5)
    assert r["all_in_range"] is True
    assert r["results"]["gamma_sig"]["in_range"] is True


def test_check_out_of_range():
    r = check_canonical_parameter_ranges(gamma_sig=1.0)
    assert r["all_in_range"] is False
    assert r["results"]["gamma_sig"]["in_range"] is False


def test_none_values_skipped():
    r = check_canonical_parameter_ranges(gamma_sig=None, theta_0=0.5)
    assert "gamma_sig" not in r["results"]
    assert "theta_0" in r["results"]


def test_empty_call_is_vacuously_in_range():
    r = check_canonical_parameter_ranges()
    assert r["all_in_range"] is True
    assert r["results"] == {}


def test_unknown_parameter_raises():
    with pytest.raises(ValueError, match="unknown canonical parameter"):
        check_canonical_parameter_ranges(not_a_real_param=1.0)


def test_default_config_is_fully_canonical():
    """The shipped defaults must sit inside every canonical range.

    Out-of-the-box behaviour is what most users will publish, so the defaults
    themselves have to be the canonical parameterisation. Previously θ₀=1.0
    (outside [0.25, 0.85]) and Π_min=0.01 (outside [0.1, 10]) meant every
    default run was non-canonical without saying so.
    """
    from config import CONFIG

    r = check_canonical_compliance_for_config(CONFIG)
    assert r["all_in_range"] is True, (
        "default CONFIG departs from canonical ranges: "
        f"{ {k: v for k, v in r['results'].items() if not v['in_range']} }"
    )
    # gamma_sig derived from ignite_tau=0.5 -> 1/0.5=2.0, within [2, 7.5]
    assert r["results"]["gamma_sig"]["in_range"] is True
    assert r["results"]["theta_0"]["in_range"] is True
    assert r["results"]["pi_min"]["in_range"] is True


def test_kappa_atp_per_bit_bands():
    assert check_canonical_parameter_ranges(kappa_atp_per_bit=100.0)["all_in_range"] is True
    assert check_canonical_parameter_ranges(kappa_atp_per_bit=5.0)["all_in_range"] is False


def test_rho_res_canonical_band():
    assert check_canonical_parameter_ranges(rho_res=0.9)["all_in_range"] is True
    assert check_canonical_parameter_ranges(rho_res=0.5)["all_in_range"] is False
