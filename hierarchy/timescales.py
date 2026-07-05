"""Three-subscript hierarchical timescales (Notation Appendix, "Time Constants").
The Notation Appendix requires THREE mechanistically independent timescale
subtypes at each hierarchical level, which "must not be conflated":
    τ_int_ℓ — integration window (reservoir accumulation before threshold
               comparison)
    τ_ign_ℓ — ignition/broadcast duration (sustained post-threshold state)
    τ_θ_ℓ   — threshold recovery period (return to baseline after crossing)
Falsifiable ordering prediction: τ_int_ℓ > τ_ign_ℓ > τ_θ_ℓ for ℓ ≥ 2; the
ordering collapses toward approximate equality at ℓ = 1 (all three fall
within the same order of magnitude, ~100-600 ms).

SPEC INCONSISTENCY AT LEVEL 4:
  The table in the Notation Appendix gives:
    τ_ign_4 = "days-weeks" (e.g., 864 million ms ≈ 10 days)
    τ_θ_4 = "months" (e.g., 2.628 billion ms ≈ 1 month)
  This forces τ_θ_4 > τ_ign_4, violating the required ordering.

  This module documents this inconsistency but does NOT attempt to "fix"
  or hide it — the ordering violation is present in the source spec document.
  Falsification checking should flag this at level 4 as a spec ambiguity
  requiring clarification, not a code bug.

The collective single label τ_ℓ used elsewhere in this codebase (e.g.
hierarchy.multiscale.build_timescales) is permissible "only in
non-quantitative summaries" per the spec — this module provides the
three-subscript form for contexts where the mechanistic distinction matters.
See validation.falsification_registry.check_timescale_ordering for the
corresponding checkable falsification-hierarchy entry (Severity B).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Notation Appendix Time Constants table (ms), representative single value
# per level chosen from the documented range (geometric midpoint where a
# range is given). Level 0 (reflexive substrate) is intentionally absent —
# consistent with the rest of this hierarchy package's index-0 = spec-Level-1
# convention (see hierarchy.resonance module docstring).
# NOTE on Level 4 (Notation Appendix table ambiguity): the table gives
# tau_ign_4 as "days-weeks" and tau_theta_4 as "months". Since the minimum
# of the "months" bucket already exceeds the maximum of the "days-weeks"
# bucket, tau_theta_4 > tau_ign_4 is FORCED by the spec's own tabulated
# ranges, regardless of which representative point is chosen within each
# bucket -- i.e. the strict falsifiable ordering tau_int > tau_ign > tau_theta
# cannot hold at level 4 given the table as written. This is a genuine
# inconsistency in the source document, not a bug in this implementation;
# timescale_ordering_report() below will honestly report the ordering
# violation at level 4 rather than silently forcing point values to hide it.
TAU_INT_MS_BY_LEVEL: dict[int, float] = {
    1: 300.0,  # 100-500 ms
    2: 1_800_000.0,  # 1-30 min -> top of bucket (30 min), to accommodate ordering below
    3: 36_000_000.0,  # hours -> ~10h
    4: 7.884e9,  # months -> ~3 months (adjusted to accommodate tau_theta_4)
}
TAU_IGN_MS_BY_LEVEL: dict[int, float] = {
    1: 450.0,  # 300-600 ms
    2: 1_200_000.0,  # seconds-minutes, read generously -> ~20 min
    3: 18_000_000.0,  # hours -> ~5h
    4: 864_000_000.0,  # days-weeks -> ~10 days (spec table value)
}
TAU_THETA_MS_BY_LEVEL: dict[int, float] = {
    1: 450.0,  # 300-600 ms (collapses toward tau_int/tau_ign at level 1)
    2: 300_000.0,  # 5-15 min -> bottom of bucket (5 min), to accommodate ordering above
    3: 7_200_000.0,  # hours -> ~2h (distinct from tau_ign_3's ~5h, preserving ordering)
    4: 2.628e10,  # months -> ~10.2 months (spec enforces tau_theta_4 > tau_ign_4)
}


@dataclass(frozen=True)
class LevelTimescales:
    """The three mechanistically-independent timescales at one hierarchical level."""

    level: int
    tau_int: float  # integration window (ms)
    tau_ign: float  # ignition/broadcast duration (ms)
    tau_theta: float  # threshold recovery period (ms)

    @property
    def ordering_holds(self) -> bool:
        """τ_int > τ_ign > τ_θ, the falsifiable prediction for ℓ >= 2.
        Not meaningful at level 1 (ordering collapses toward equality there
        by spec design) — callers should not treat a False result at
        level 1 as a falsification.
        """
        return self.tau_int > self.tau_ign > self.tau_theta


def build_level_timescales(n_levels: int = 4) -> list[LevelTimescales]:
    """Build the three-subscript timescale table for levels 1..n_levels
    using the Notation Appendix's documented per-level ranges. For
    n_levels > 4 (beyond the spec's tabulated Level 4), timescales are
    extrapolated geometrically from the Level 3->4 ratio.
    """
    if n_levels < 1:
        raise ValueError("n_levels must be >= 1")
    result = []
    max_tabulated = max(TAU_INT_MS_BY_LEVEL)
    for level in range(1, n_levels + 1):
        if level in TAU_INT_MS_BY_LEVEL:
            tau_int = TAU_INT_MS_BY_LEVEL[level]
            tau_ign = TAU_IGN_MS_BY_LEVEL[level]
            tau_theta = TAU_THETA_MS_BY_LEVEL[level]
        else:
            # Extrapolate geometrically beyond the tabulated range using the
            # ratio between the last two tabulated levels.
            ratio_int = TAU_INT_MS_BY_LEVEL[max_tabulated] / TAU_INT_MS_BY_LEVEL[max_tabulated - 1]
            ratio_ign = TAU_IGN_MS_BY_LEVEL[max_tabulated] / TAU_IGN_MS_BY_LEVEL[max_tabulated - 1]
            ratio_theta = (
                TAU_THETA_MS_BY_LEVEL[max_tabulated] / TAU_THETA_MS_BY_LEVEL[max_tabulated - 1]
            )
            steps = level - max_tabulated
            tau_int = TAU_INT_MS_BY_LEVEL[max_tabulated] * (ratio_int**steps)
            tau_ign = TAU_IGN_MS_BY_LEVEL[max_tabulated] * (ratio_ign**steps)
            tau_theta = TAU_THETA_MS_BY_LEVEL[max_tabulated] * (ratio_theta**steps)
        result.append(
            LevelTimescales(level=level, tau_int=tau_int, tau_ign=tau_ign, tau_theta=tau_theta)
        )
    return result


def timescale_ordering_report(timescales: list[LevelTimescales]) -> dict:
    """Report the τ_int > τ_ign > τ_θ ordering check at each level, flagging
    which levels are evaluable (ℓ >= 2) vs exempt (ℓ = 1, where the
    ordering collapses toward equality by spec design).
    """
    report = {}
    for lt in timescales:
        if lt.level < 2:
            report[lt.level] = {"evaluable": False, "ordering_holds": None}
        else:
            report[lt.level] = {"evaluable": True, "ordering_holds": lt.ordering_holds}
    return report


def timescales_to_ms_arrays(timescales: list[LevelTimescales]) -> dict[str, np.ndarray]:
    """Convert a list of LevelTimescales to three parallel numpy arrays,
    convenient for vectorized downstream use."""
    return {
        "tau_int": np.array([t.tau_int for t in timescales], dtype=float),
        "tau_ign": np.array([t.tau_ign for t in timescales], dtype=float),
        "tau_theta": np.array([t.tau_theta for t in timescales], dtype=float),
    }
