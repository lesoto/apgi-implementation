"""Tests for hierarchy.timescales and hierarchy.criticality (three-subscript
timescales and per-level rho_crit, Notation Appendix / Dynamical Formulation §10)."""

import numpy as np
import pytest

from hierarchy.criticality import (
    HierarchicalCriticalityTracker,
    LevelRecurrentField,
    compute_rho_crit,
    make_recurrent_matrix,
)
from hierarchy.timescales import (
    LevelTimescales,
    build_level_timescales,
    timescale_ordering_report,
    timescales_to_ms_arrays,
)


class TestLevelTimescales:
    def test_build_default_four_levels(self):
        ts = build_level_timescales(4)
        assert len(ts) == 4
        assert [t.level for t in ts] == [1, 2, 3, 4]

    def test_ordering_holds_for_levels_2_and_3(self):
        ts = build_level_timescales(4)
        by_level = {t.level: t for t in ts}
        assert by_level[2].ordering_holds is True
        assert by_level[3].ordering_holds is True

    def test_level_1_not_asserted_to_hold(self):
        # Level 1 collapses toward equality by spec design; not a failure.
        ts = build_level_timescales(1)
        assert ts[0].level == 1

    def test_invalid_n_levels_raises(self):
        with pytest.raises(ValueError):
            build_level_timescales(0)

    def test_extrapolation_beyond_tabulated_levels(self):
        ts = build_level_timescales(6)
        assert len(ts) == 6
        # Extrapolated levels should keep tau_int increasing monotonically
        taus_int = [t.tau_int for t in ts]
        assert taus_int == sorted(taus_int)


class TestTimescaleOrderingReport:
    def test_level_1_marked_not_evaluable(self):
        ts = build_level_timescales(4)
        report = timescale_ordering_report(ts)
        assert report[1]["evaluable"] is False
        assert report[1]["ordering_holds"] is None

    def test_levels_2_and_3_evaluable_and_hold(self):
        ts = build_level_timescales(4)
        report = timescale_ordering_report(ts)
        assert report[2]["evaluable"] is True
        assert report[2]["ordering_holds"] is True
        assert report[3]["ordering_holds"] is True

    def test_level_4_evaluable_but_documented_exception(self):
        """Level 4 legitimately fails the ordering given the spec's own
        table (months > days-weeks) — this is a documented source-table
        inconsistency, not an implementation bug."""
        ts = build_level_timescales(4)
        report = timescale_ordering_report(ts)
        assert report[4]["evaluable"] is True
        assert report[4]["ordering_holds"] is False


def test_timescales_to_ms_arrays():
    ts = build_level_timescales(3)
    arrays = timescales_to_ms_arrays(ts)
    assert set(arrays.keys()) == {"tau_int", "tau_ign", "tau_theta"}
    assert len(arrays["tau_int"]) == 3


def test_falsification_registry_integration():
    """The three-subscript timescales here should be usable directly with
    validation.falsification_registry.check_timescale_ordering."""
    from validation.falsification_registry import check_timescale_ordering

    ts = build_level_timescales(4)
    by_level = {t.level: t for t in ts}
    r = check_timescale_ordering(
        by_level[2].tau_int, by_level[2].tau_ign, by_level[2].tau_theta, level=2
    )
    assert r["verdict"] == "confirmed"


class TestComputeRhoCrit:
    def test_rho_crit_of_scaled_identity(self):
        W = np.eye(5) * 0.7
        assert compute_rho_crit(W) == pytest.approx(0.7)

    def test_non_square_raises(self):
        with pytest.raises(ValueError):
            compute_rho_crit(np.ones((3, 4)))


class TestMakeRecurrentMatrix:
    def test_achieves_target_rho_crit(self):
        rng = np.random.default_rng(0)
        W = make_recurrent_matrix(30, target_rho_crit=1.0, density=0.3, rng=rng)
        assert compute_rho_crit(W) == pytest.approx(1.0, abs=1e-6)

    def test_different_target(self):
        rng = np.random.default_rng(0)
        W = make_recurrent_matrix(20, target_rho_crit=0.5, rng=rng)
        assert compute_rho_crit(W) == pytest.approx(0.5, abs=1e-6)

    def test_invalid_n_units(self):
        with pytest.raises(ValueError):
            make_recurrent_matrix(0)


class TestLevelRecurrentField:
    def test_is_near_critical(self):
        field = LevelRecurrentField(level=1, W_rec=np.eye(5) * 1.02)
        assert field.rho_crit == pytest.approx(1.02)
        assert field.is_near_critical(tolerance=0.1) is True
        assert field.is_near_critical(tolerance=0.01) is False


class TestHierarchicalCriticalityTracker:
    def test_tracks_all_levels(self):
        tracker = HierarchicalCriticalityTracker(
            n_levels=4, n_units_per_level=15, rng=np.random.default_rng(1)
        )
        rho_by_level = tracker.rho_crit_by_level()
        assert set(rho_by_level.keys()) == {1, 2, 3, 4}
        for rho in rho_by_level.values():
            assert rho == pytest.approx(1.0, abs=1e-5)

    def test_report_structure(self):
        tracker = HierarchicalCriticalityTracker(
            n_levels=3, n_units_per_level=10, rng=np.random.default_rng(2)
        )
        report = tracker.report()
        assert "per_level" in report
        assert "fraction_near_critical" in report
        assert report["fraction_near_critical"] == 1.0

    def test_rho_res_and_rho_crit_are_distinct_concepts(self):
        """Sanity check: this module's rho_crit targets ~1.0 (criticality),
        NOT the reservoir's sub-critical rho_res in [0.7, 0.95]."""
        tracker = HierarchicalCriticalityTracker(
            n_levels=1, n_units_per_level=10, target_rho_crit=1.0, rng=np.random.default_rng(3)
        )
        rho = tracker.rho_crit_by_level()[1]
        assert not (0.7 <= rho <= 0.95)  # not accidentally in the reservoir's sub-critical band
        assert rho == pytest.approx(1.0, abs=1e-5)

    def test_invalid_n_levels_raises(self):
        with pytest.raises(ValueError):
            HierarchicalCriticalityTracker(n_levels=0)
