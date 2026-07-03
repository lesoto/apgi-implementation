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


def test_config_for_default_config_reports_deviations():
    from config import CONFIG

    r = check_canonical_compliance_for_config(CONFIG)
    # gamma_sig derived from ignite_tau=0.5 -> 1/0.5=2.0, within [2, 7.5]
    assert r["results"]["gamma_sig"]["in_range"] is True
    # theta_base default (1.0) is a documented non-canonical lightweight value
    assert r["results"]["theta_0"]["in_range"] is False


def test_kappa_atp_per_bit_bands():
    assert check_canonical_parameter_ranges(kappa_atp_per_bit=100.0)["all_in_range"] is True
    assert check_canonical_parameter_ranges(kappa_atp_per_bit=5.0)["all_in_range"] is False


def test_rho_res_canonical_band():
    assert check_canonical_parameter_ranges(rho_res=0.9)["all_in_range"] is True
    assert check_canonical_parameter_ranges(rho_res=0.5)["all_in_range"] is False
