"""Tests for validation.falsification_registry (Dynamical Formulation §27)."""

import numpy as np
import pytest

from stats.avalanche import fit_discrete_power_law_mle
from validation.falsification_registry import (
    FALSIFICATION_TABLE,
    check_criticality_avalanche,
    check_discrete_ignition,
    check_long_range_correlations,
    check_oscillatory_gate,
    check_phase_transition_signatures,
    check_thermodynamic_bridge,
    check_timescale_ordering,
    check_valence_asymmetry,
    get_claim,
    summarize_registry,
)


def test_table_has_all_severities():
    severities = {c.severity for c in FALSIFICATION_TABLE}
    assert severities == {"A", "B", "C"}
    ids = [c.claim_id for c in FALSIFICATION_TABLE]
    assert len(ids) == len(set(ids)), "duplicate claim_id"


def test_get_claim_roundtrip():
    claim = get_claim("discrete_ignition")
    assert claim.severity == "A"
    with pytest.raises(KeyError):
        get_claim("nonexistent")


class TestDiscreteIgnition:
    def test_falsified_below_5(self):
        r = check_discrete_ignition(gamma_sig=5.0, alpha_psy=3.0)
        assert r["verdict"] == "falsified"
        assert r["severity"] == "A"

    def test_confirmed_at_or_above_10(self):
        r = check_discrete_ignition(gamma_sig=5.0, alpha_psy=12.0)
        assert r["verdict"] == "confirmed"

    def test_indeterminate_without_alpha_psy(self):
        r = check_discrete_ignition(gamma_sig=5.0)
        assert r["verdict"] == "indeterminate"

    def test_indeterminate_in_gap(self):
        r = check_discrete_ignition(gamma_sig=5.0, alpha_psy=7.0)
        assert r["verdict"] == "indeterminate"


class TestLongRangeCorrelations:
    def test_falsified_below_threshold(self):
        r = check_long_range_correlations(alpha_dfa=0.4, n_trials=300)
        assert r["verdict"] == "falsified"

    def test_confirmed_full_hierarchy(self):
        r = check_long_range_correlations(alpha_dfa=0.9, n_trials=300)
        assert r["verdict"] == "confirmed"

    def test_confirmed_partial_hierarchy(self):
        r = check_long_range_correlations(alpha_dfa=0.70, n_trials=300, partial_hierarchy=True)
        assert r["verdict"] == "confirmed"

    def test_insufficient_trials_indeterminate(self):
        r = check_long_range_correlations(alpha_dfa=0.9, n_trials=50)
        assert r["verdict"] == "indeterminate"


class TestThermodynamicBridge:
    def test_consistent(self):
        r = check_thermodynamic_bridge(100.0)
        assert r["verdict"] == "confirmed"

    def test_requires_explanation(self):
        r = check_thermodynamic_bridge(5000.0)
        assert r["verdict"] == "requires_explanation"

    def test_bridge_inconsistent(self):
        r = check_thermodynamic_bridge(5.0)
        assert r["verdict"] == "falsified"


class TestValenceAsymmetry:
    def test_symmetric_default_indeterminate(self):
        r = check_valence_asymmetry(1.0, 1.0)
        assert r["verdict"] == "indeterminate"

    def test_asymmetric_confirmed(self):
        r = check_valence_asymmetry(1.5, 0.8)
        assert r["verdict"] == "confirmed"

    def test_symmetric_with_strong_bf_falsified(self):
        r = check_valence_asymmetry(1.0, 1.0, bf10=20.0, n=50)
        assert r["verdict"] == "falsified"


class TestTimescaleOrdering:
    def test_level1_indeterminate(self):
        r = check_timescale_ordering(10, 5, 1, level=1)
        assert r["verdict"] == "indeterminate"

    def test_ordered_confirmed(self):
        r = check_timescale_ordering(10, 5, 1, level=3)
        assert r["verdict"] == "confirmed"

    def test_unordered_falsified(self):
        r = check_timescale_ordering(1, 5, 10, level=3)
        assert r["verdict"] == "falsified"


class TestOscillatoryGate:
    def test_confirmed_large_effect(self):
        rng = np.random.default_rng(0)
        gc = rng.normal(0.7, 0.05, 40)
        gm = rng.normal(0.3, 0.05, 40)
        r = check_oscillatory_gate(gc, gm)
        assert r["verdict"] == "confirmed"

    def test_falsified_no_effect(self):
        rng = np.random.default_rng(1)
        gc = rng.normal(0.5, 0.05, 40)
        gm = rng.normal(0.5, 0.05, 40)
        r = check_oscillatory_gate(gc, gm)
        assert r["verdict"] == "falsified"

    def test_indeterminate_below_n_min(self):
        r = check_oscillatory_gate([0.6, 0.7], [0.3, 0.2])
        assert r["verdict"] == "indeterminate"


class TestPhaseTransitionSignatures:
    def test_insufficient_samples(self):
        r = check_phase_transition_signatures(np.zeros(10), theta=1.0)
        assert r["verdict"] == "indeterminate"

    def test_returns_signature_count(self):
        rng = np.random.default_rng(2)
        S = np.concatenate([rng.normal(0.5, 0.1, 200), rng.normal(3.0, 1.0, 200)])
        r = check_phase_transition_signatures(S, theta=1.0)
        assert r["verdict"] in ("confirmed", "falsified", "indeterminate")
        assert 0 <= r["n_signatures_confirmed"] <= 3


class TestCriticalityAvalanche:
    @staticmethod
    def _activity_from_sizes(sizes) -> np.ndarray:
        parts: list[int] = []
        for s in sizes:
            parts.extend([1] * int(s))
            parts.append(0)
        return np.array(parts)

    def test_insufficient_data_indeterminate(self):
        r = check_criticality_avalanche(np.zeros(10), n_participants=20)
        assert r["verdict"] == "indeterminate"
        assert r["severity"] == "A"

    def test_below_n_participants_indeterminate(self):
        rng = np.random.default_rng(42)
        u = rng.uniform(0, 1, size=500)
        x = np.clip(np.floor((1 - u) ** (-1 / (1.5 - 1))), 1, 200)
        activity = self._activity_from_sizes(x)
        r = check_criticality_avalanche(activity, n_participants=5)
        assert r["verdict"] == "indeterminate"

    def test_true_power_law_confirmed(self):
        """Sizes drawn from a true alpha=1.5 power law (inverse-transform
        sampling) should fit within the tolerance band and confirm."""
        rng = np.random.default_rng(42)
        u = rng.uniform(0, 1, size=500)
        x = np.clip(np.floor((1 - u) ** (-1 / (1.5 - 1))), 1, 200)
        activity = self._activity_from_sizes(x)
        r = check_criticality_avalanche(activity, n_participants=20)
        assert r["verdict"] == "confirmed"
        assert 1.2 <= r["alpha"] <= 1.8

    def test_exponential_subcritical_falsified(self, monkeypatch):
        """A fitted exponent above the achievable-range's exponential
        threshold (concentrated, short avalanches) falsifies the claim."""
        canned = {
            "status": "success",
            "n_avalanches": 200,
            "alpha": 2.35,
            "alpha_target": 1.5,
            "tolerance": 0.3,
            "within_tolerance": False,
            "xmin": 1,
        }
        monkeypatch.setattr("stats.avalanche.validate_avalanche_power_law", lambda *a, **k: canned)
        r = check_criticality_avalanche(np.ones(50), n_participants=20)
        assert r["verdict"] == "falsified"
        assert r["alpha"] > r["exponential_threshold"]

    def test_flat_supercritical_falsified(self, monkeypatch):
        """A fitted exponent near the estimator's theoretical floor (broad,
        near-uniform avalanche sizes) falsifies the claim."""
        canned = {
            "status": "success",
            "n_avalanches": 300,
            "alpha": 1.05,
            "alpha_target": 1.5,
            "tolerance": 0.3,
            "within_tolerance": False,
            "xmin": 1,
        }
        monkeypatch.setattr("stats.avalanche.validate_avalanche_power_law", lambda *a, **k: canned)
        r = check_criticality_avalanche(np.ones(50), n_participants=20)
        assert r["verdict"] == "falsified"
        assert r["alpha"] < r["flat_threshold"]

    def test_mild_deviation_indeterminate(self, monkeypatch):
        """A deviation outside the tolerance band but not extreme enough to
        reach either falsifying extreme is indeterminate, not falsified."""
        canned = {
            "status": "success",
            "n_avalanches": 100,
            "alpha": 1.9,
            "alpha_target": 1.5,
            "tolerance": 0.3,
            "within_tolerance": False,
            "xmin": 1,
        }
        monkeypatch.setattr("stats.avalanche.validate_avalanche_power_law", lambda *a, **k: canned)
        r = check_criticality_avalanche(np.ones(50), n_participants=20)
        assert r["verdict"] == "indeterminate"

    def test_estimator_bound_sanity(self):
        """Sanity check on the documented achievable range for xmin=1: the
        discrete MLE cannot exceed 1 + 1/ln(2)."""
        sizes_all_singleton = np.ones(500, dtype=int)
        fit = fit_discrete_power_law_mle(sizes_all_singleton, xmin=1)
        assert fit.alpha == pytest.approx(1 + 1 / np.log(2), abs=1e-9)


def test_summarize_registry():
    results = [
        check_thermodynamic_bridge(100.0),
        check_thermodynamic_bridge(5.0),
    ]
    summary = summarize_registry(results)
    assert summary["tally"]["B"]["confirmed"] == 1
    assert summary["tally"]["B"]["falsified"] == 1
    assert summary["any_severity_a_falsified"] is False
