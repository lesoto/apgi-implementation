"""Per-level criticality regulation — ρ_crit^(l) (Dynamical Formulation §10).
    ρ_crit^(l) = ρ(W_rec^(l))    [spectral radius of the recurrent weight
                                   matrix at hierarchical level l]
    Optimal ignition: ρ_crit^(l) ≈ 1.0
This is architecturally DISTINCT from the reservoir substrate's ρ_res
(Notation Appendix: "Reserve ρ_crit exclusively for the criticality regime
ρ_crit ≈ 1 (recurrent attractor field W_rec)"; ρ_res ∈ [0.7, 0.95] governs
the sub-critical liquid-state-machine reservoir, a different substrate).
W_rec^(l) is the recurrent attractor-field weight matrix at level l (§9's
"Recurrent Competitive Attractor Dynamics"), previously absent from this
codebase entirely — no per-level recurrent matrix or spectral-radius
tracking existed anywhere. This module provides that missing structure as a
standalone, importable component; it is not wired into APGIPipeline's
per-step loop by default (the full recurrent competitive-dynamics subsystem
of Dynamical Formulation §9 — winner-take-all competition, top-down/
bottom-up coupling, divisive inhibition — is a substantially larger
addition than the criticality bookkeeping alone).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def compute_rho_crit(W_rec: np.ndarray) -> float:
    """ρ_crit = ρ(W_rec): the spectral radius (largest eigenvalue magnitude)
    of a recurrent weight matrix.
    """
    W = np.asarray(W_rec, dtype=float)
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError(f"W_rec must be square, got shape {W.shape}")
    with np.errstate(all="ignore"):
        eigs = np.linalg.eigvals(W)
    return float(np.max(np.abs(eigs)))


def make_recurrent_matrix(
    n_units: int,
    target_rho_crit: float = 1.0,
    density: float = 0.2,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Construct a random sparse recurrent matrix W_rec normalized to a
    target spectral radius (default 1.0, the spec's optimal-ignition
    criticality point).
    """
    if n_units <= 0:
        raise ValueError("n_units must be > 0")
    generator = rng or np.random.default_rng()
    mask = (generator.random((n_units, n_units)) < density).astype(float)
    W_raw = generator.normal(size=(n_units, n_units)) * mask
    rho_raw = compute_rho_crit(W_raw)
    rho_raw = max(rho_raw, 1e-8)
    result: np.ndarray = W_raw * (target_rho_crit / rho_raw)
    return result


@dataclass
class LevelRecurrentField:
    """The recurrent attractor field W_rec^(l) at one hierarchical level,
    with its criticality diagnostics.
    """

    level: int
    W_rec: np.ndarray

    @property
    def rho_crit(self) -> float:
        return compute_rho_crit(self.W_rec)

    def is_near_critical(self, tolerance: float = 0.1) -> bool:
        """True if ρ_crit is within `tolerance` of the spec's optimal-
        ignition point ρ_crit ≈ 1.0."""
        return bool(abs(self.rho_crit - 1.0) <= tolerance)


@dataclass
class HierarchicalCriticalityTracker:
    """Per-level ρ_crit^(l) tracking across the hierarchy.
    Distinct from the reservoir's single scalar ρ_res (sub-critical,
    [0.7, 0.95]) — each hierarchical level here gets its own independent
    recurrent field and criticality measurement.
    """

    n_levels: int
    n_units_per_level: int = 20
    target_rho_crit: float = 1.0
    density: float = 0.2
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng())
    fields: list[LevelRecurrentField] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if self.n_levels < 1:
            raise ValueError("n_levels must be >= 1")
        self.fields = [
            LevelRecurrentField(
                level=level,
                W_rec=make_recurrent_matrix(
                    self.n_units_per_level, self.target_rho_crit, self.density, self.rng
                ),
            )
            for level in range(1, self.n_levels + 1)
        ]

    def rho_crit_by_level(self) -> dict[int, float]:
        """ρ_crit^(l) for every tracked level."""
        return {lf.level: lf.rho_crit for lf in self.fields}

    def report(self, tolerance: float = 0.1) -> dict:
        """Structured criticality report: per-level ρ_crit, near-critical
        flag, and the fraction of levels currently near criticality.
        """
        per_level = {
            lf.level: {
                "rho_crit": lf.rho_crit,
                "near_critical": lf.is_near_critical(tolerance),
            }
            for lf in self.fields
        }
        fraction_critical = float(np.mean([v["near_critical"] for v in per_level.values()]))
        return {"per_level": per_level, "fraction_near_critical": fraction_critical}
