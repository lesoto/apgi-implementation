"""APGI Falsification Hierarchy Registry (Dynamical Formulation §27).
Implements the pre-registered falsification hierarchy as CHECKABLE code —
not just documentation. Per §27, falsification criteria are organized into
three severities:
    Severity A — Framework-disconfirming (core claim abandoned)
    Severity B — Revision-requiring (component modified, core intact)
    Severity C — Peripheral (secondary predictions; null results leave the
                 core framework intact)
The FALSIFICATION_TABLE below reproduces the full pre-registered table
(claim, quantitative prediction, falsifying result, severity) as structured
data for every row, including the purely-empirical claims (e.g. allostatic
threshold response to metabolic challenge) that cannot be evaluated from
simulation state alone — those rows carry `checkable=False` and exist so the
complete registry is inspectable in one place, per the Notation Appendix's
insistence that canonical numeric bounds not live only in prose.
For the claims that ARE computable from model/simulation state, `check_*`
functions implement the actual pass/fail/indeterminate logic using the exact
thresholds from the spec.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from energy.thermodynamics import classify_kappa_consistency

Severity = Literal["A", "B", "C"]


@dataclass(frozen=True)
class FalsificationClaim:
    """One row of the pre-registered falsification hierarchy (§27)."""

    claim_id: str
    description: str
    quantitative_prediction: str
    falsifying_result: str
    severity: Severity
    checkable: bool
    section: str


FALSIFICATION_TABLE: tuple[FalsificationClaim, ...] = (
    FalsificationClaim(
        claim_id="discrete_ignition",
        description="Discrete ignition (model-internal sigmoid + behavioural steepness)",
        quantitative_prediction="gamma_sig = 1/tau_sigma in [2, 7.5]; alpha_psy >= 10",
        falsifying_result="Best-fitting alpha_psy < 5 across all tested paradigms, "
        "or a graded-accumulation model wins by delta-AIC > 10",
        severity="A",
        checkable=True,
        section="§15 / §27",
    ),
    FalsificationClaim(
        claim_id="allostatic_threshold",
        description="Allostatic threshold modulated by metabolic/circadian/caloric state",
        quantitative_prediction="theta_t modulated by glucose depletion, circadian phase, "
        "and caloric restriction after controlling for arousal",
        falsifying_result="theta_t invariant across all three manipulations in a "
        "well-powered sample (d > 0.5, n >= 30, pre-registered)",
        severity="A",
        checkable=False,
        section="§14 / §27",
    ),
    FalsificationClaim(
        claim_id="precision_ignition_link",
        description="P3b amplitude tracks precision-weighted salience",
        quantitative_prediction="P3b amplitude tracks Pi * |phi(epsilon)|",
        falsifying_result="No relationship across >= 1000 trials, N >= 30, with "
        "reliable precision manipulation",
        severity="A",
        checkable=False,
        section="§27",
    ),
    FalsificationClaim(
        claim_id="computational_benefit",
        description="APGI-LNN outperforms an additive GNW model",
        quantitative_prediction="APGI-LNN beats additive GNW model, BF > 100",
        falsifying_result="BF < 100 in a pre-registered dataset (N >= 500 trials x >= 30 participants)",
        severity="A",
        checkable=False,
        section="§27",
    ),
    FalsificationClaim(
        claim_id="long_range_correlations",
        description="1/f threshold dynamics — long-range temporal correlations",
        quantitative_prediction="H = alpha_DFA ~ 0.85-0.95 (fGn) in threshold/detection-rate "
        "series; partial-hierarchy (L0-L2) H ~ 0.65-0.75",
        falsifying_result="alpha_DFA < 0.55 across two independent datasets (DFA, >= 200 trials)",
        severity="A",
        checkable=True,
        section="§12 / §21.1 / §27",
    ),
    FalsificationClaim(
        claim_id="criticality_avalanche",
        description="Neural avalanche size distributions follow power-law scaling "
        "at conscious ignition (ρ_crit ≈ 1.0)",
        quantitative_prediction="Avalanche size distribution exponent ≈ -1.5 "
        "(Beggs & Plenz, 2003) during verified conscious access",
        falsifying_result="Avalanche distributions are exponential (sub-critical) "
        "or flat (super-critical) during verified conscious access across "
        "N >= 20 participants",
        severity="A",
        checkable=True,
        section="§22.1 / §27",
    ),
    FalsificationClaim(
        claim_id="valence_asymmetry",
        description="Signed prediction error valence asymmetry",
        quantitative_prediction="alpha_plus / alpha_minus != 1 in an appetitive-vs-aversive "
        "near-threshold task",
        falsifying_result="Asymmetry undetectable; BF > 10 for the symmetric model, n >= 40 "
        "-> revert to the |epsilon| formulation",
        severity="B",
        checkable=True,
        section="§3 (Dynamical) / §27",
    ),
    FalsificationClaim(
        claim_id="thermodynamic_bridge",
        description="Metabolic bridge parameter kappa consistency",
        quantitative_prediction="kappa in [10, 1000] ATP/bit (supported); "
        "(1000, 10000] requires explanation",
        falsifying_result="kappa < 10 or kappa > 10000 by PET / 31P-MRS calibration "
        "-> thermodynamic bridge falsified",
        severity="B",
        checkable=True,
        section="§11 / §13 (Dynamical) / §27",
    ),
    FalsificationClaim(
        claim_id="phase_transition_signatures",
        description="Three simultaneous phase-transition signatures at ignition",
        quantitative_prediction="Bistability (Cohen's d > 0.8), critical slowing (> 20%), "
        "and elevated susceptibility (> 1.5x) co-occur at ignition",
        falsifying_result="Only 1 of 3 signatures confirmed across two datasets "
        "-> revise to a graded-transition model",
        severity="B",
        checkable=True,
        section="§10 / §27",
    ),
    FalsificationClaim(
        claim_id="oscillatory_gate",
        description="Oscillatory synchrony gate distinguishes conscious access",
        quantitative_prediction="Inter-level Gamma coherence is higher on conscious than "
        "near-miss trials (d > 0.5, n >= 30)",
        falsifying_result="Gamma statistically equivalent across two independent datasets "
        "-> absorb into the PAC precision term",
        severity="B",
        checkable=True,
        section="§8 / §21.2 / §27",
    ),
    FalsificationClaim(
        claim_id="serotonin_mapping",
        description="Serotonin additive threshold offset dissociates from NE",
        quantitative_prediction="SSRIs shift theta_eff additively; NE/atomoxetine shifts "
        "theta multiplicatively (dissociable pathways)",
        falsifying_result="5-HT manipulation fails to shift theta_eff in the predicted direction",
        severity="B",
        checkable=False,
        section="§7.4 (Dynamical) / §27",
    ),
    FalsificationClaim(
        claim_id="reservoir_advantage",
        description="Liquid-network reservoir outperforms RNN/feedforward baselines",
        quantitative_prediction="LNN outperforms RNN / feed-forward on the temporal-"
        "integration battery without augmentation",
        falsifying_result="Augmented alternatives match LNN performance (n-sim > 100 "
        "per architecture)",
        severity="B",
        checkable=False,
        section="§17 / §27",
    ),
    FalsificationClaim(
        claim_id="timescale_ordering",
        description="Three-timescale ordering within each hierarchical level",
        quantitative_prediction="tau_int,l > tau_ign,l > tau_theta,l for l >= 2 "
        "(ordering collapses toward equality at l = 1)",
        falsifying_result="Pharmacological dissociation fails to separate the three timescales",
        severity="B",
        checkable=True,
        section="Notation Appendix (Time Constants) / §27",
    ),
    FalsificationClaim(
        claim_id="metabolic_specificity",
        description="Peripheral (non-cerebral) metabolic perturbation spares theta_t",
        quantitative_prediction="A peripheral perturbation that spares CMRglc does NOT raise theta_t",
        falsifying_result="An equivalent non-cerebral metabolic perturbation raises "
        "theta_t (insulin-clamp dissociation)",
        severity="B",
        checkable=False,
        section="§13 (Dynamical) / §27",
    ),
    FalsificationClaim(
        claim_id="cross_species_circadian_ultradian",
        description="Cross-species PCI gradient; circadian/ultradian theta_t modulation",
        quantitative_prediction="Cross-species PCI gradient; circadian and ~90-min "
        "ultradian theta_t modulation",
        falsifying_result="Null at current sample sizes — core framework intact",
        severity="C",
        checkable=False,
        section="§27",
    ),
)


def get_claim(claim_id: str) -> FalsificationClaim:
    """Look up a single claim by id."""
    for claim in FALSIFICATION_TABLE:
        if claim.claim_id == claim_id:
            return claim
    raise KeyError(f"unknown falsification claim_id: {claim_id!r}")


#: The four verdicts a falsification claim can receive. Named so call sites can
#: annotate their local `verdict` variable instead of widening to bare `str`,
#: which would let a typo through into a reported falsification outcome.
Verdict = Literal["confirmed", "falsified", "requires_explanation", "indeterminate"]


def _result(
    claim_id: str,
    verdict: Verdict,
    detail: dict[str, Any],
) -> dict[str, Any]:
    claim = get_claim(claim_id)
    return {
        "claim_id": claim_id,
        "severity": claim.severity,
        "verdict": verdict,
        "section": claim.section,
        **detail,
    }


def check_discrete_ignition(
    gamma_sig: float,
    alpha_psy: float | None = None,
) -> dict[str, Any]:
    """Severity A: γ_sig ∈ [2, 7.5] (model-internal band); α_psy >= 10
    supported, α_psy < 5 falsifies discrete ignition (behavioural signature).
    Args:
        gamma_sig: Model-internal ignition-sigmoid steepness (= 1/τ_σ)
        alpha_psy: Behavioural psychometric steepness, if measured. If None,
            only the model-internal band is checked and the verdict is
            "indeterminate" (the falsifiable behavioural test requires data).
    """
    gamma_in_band = 2.0 <= gamma_sig <= 7.5
    if alpha_psy is None:
        return _result(
            "discrete_ignition",
            "indeterminate",
            {
                "gamma_sig": gamma_sig,
                "gamma_sig_in_band": gamma_in_band,
                "alpha_psy": None,
                "note": "alpha_psy not provided; behavioural falsifier requires empirical data",
            },
        )
    verdict: Verdict
    if alpha_psy < 5.0:
        verdict = "falsified"
    elif alpha_psy >= 10.0:
        verdict = "confirmed"
    else:
        verdict = "indeterminate"
    return _result(
        "discrete_ignition",
        verdict,
        {
            "gamma_sig": gamma_sig,
            "gamma_sig_in_band": gamma_in_band,
            "alpha_psy": alpha_psy,
        },
    )


def check_long_range_correlations(
    alpha_dfa: float,
    n_trials: int,
    partial_hierarchy: bool = False,
) -> dict[str, Any]:
    """Severity A: H = α_DFA ≈ 0.85-0.95 (full hierarchy, fGn) or ≈ 0.65-0.75
    (partial hierarchy L0-L2); falsifier α_DFA < 0.55 across >= 200 trials.
    """
    if n_trials < 200:
        return _result(
            "long_range_correlations",
            "indeterminate",
            {
                "alpha_dfa": alpha_dfa,
                "n_trials": n_trials,
                "note": "spec requires >= 200 trials for a confirmatory DFA estimate",
            },
        )
    verdict: Verdict
    if alpha_dfa < 0.55:
        verdict = "falsified"
    else:
        lo, hi = (0.65, 0.75) if partial_hierarchy else (0.85, 0.95)
        verdict = "confirmed" if lo <= alpha_dfa <= hi else "indeterminate"
    return _result(
        "long_range_correlations",
        verdict,
        {
            "alpha_dfa": alpha_dfa,
            "n_trials": n_trials,
            "partial_hierarchy": partial_hierarchy,
            "predicted_range": (0.65, 0.75) if partial_hierarchy else (0.85, 0.95),
        },
    )


def check_criticality_avalanche(
    activity: np.ndarray,
    n_participants: int = 1,
    alpha_target: float = 1.5,
    tolerance: float = 0.3,
    exponential_threshold: float = 2.0,
    flat_threshold: float = 1.1,
    xmin: int = 1,
    min_avalanches: int = 30,
) -> dict[str, Any]:
    """Severity A: avalanche size distribution exponent ~ 1.5 during conscious
    access (§22.1). Falsifier: distribution is exponential/sub-critical
    (fitted exponent far steeper than the power-law target, > exponential_threshold)
    or flat/super-critical (exponent near zero, < flat_threshold) across
    N >= 20 participants. A fit within [alpha_target - tolerance,
    alpha_target + tolerance] confirms; anything between the tolerance band
    and the exponential/flat extremes is indeterminate (deviation present but
    not extreme enough to constitute the pre-registered falsifying result).
    Delegates the underlying fit to stats.avalanche.validate_avalanche_power_law.
    Note: the discrete-MLE estimator used here (xmin=1, standard Clauset-style
    approximation) is mathematically bounded to alpha_hat in [1, 1 + 1/ln(2)]
    ≈ [1.0, 2.44] — exponential_threshold and flat_threshold are calibrated to
    sit within that achievable range rather than at the nominal 0/∞ extremes a
    naive reading of "exponential" / "flat" might suggest.
    """
    from stats.avalanche import validate_avalanche_power_law

    fit = validate_avalanche_power_law(
        activity,
        alpha_target=alpha_target,
        tolerance=tolerance,
        xmin=xmin,
        min_avalanches=min_avalanches,
    )
    if fit["status"] == "insufficient_data":
        return _result(
            "criticality_avalanche",
            "indeterminate",
            {**fit, "n_participants": n_participants, "note": "insufficient avalanches for a fit"},
        )
    alpha = fit["alpha"]
    verdict: Verdict
    if n_participants < 20:
        verdict = "indeterminate"
    elif fit["within_tolerance"]:
        verdict = "confirmed"
    elif alpha > exponential_threshold or alpha < flat_threshold:
        verdict = "falsified"
    else:
        verdict = "indeterminate"
    return _result(
        "criticality_avalanche",
        verdict,
        {
            **fit,
            "n_participants": n_participants,
            "exponential_threshold": exponential_threshold,
            "flat_threshold": flat_threshold,
        },
    )


def check_valence_asymmetry(
    alpha_plus: float,
    alpha_minus: float,
    bf10: float | None = None,
    n: int | None = None,
) -> dict[str, Any]:
    """Severity B: α⁺/α⁻ ≠ 1 required; falsifier BF₁₀ > 10 for the symmetric
    model (i.e. BF favouring symmetry) with n >= 40.
    """
    ratio = alpha_plus / alpha_minus if alpha_minus != 0 else float("inf")
    asymmetric = not np.isclose(ratio, 1.0, atol=1e-9)
    verdict: Verdict
    if bf10 is not None and n is not None and n >= 40 and bf10 > 10 and not asymmetric:
        verdict = "falsified"
    elif asymmetric:
        verdict = "confirmed"
    else:
        verdict = "indeterminate"
    return _result(
        "valence_asymmetry",
        verdict,
        {
            "alpha_plus": alpha_plus,
            "alpha_minus": alpha_minus,
            "ratio": ratio,
            "bf10": bf10,
            "n": n,
        },
    )


def check_thermodynamic_bridge(kappa: float) -> dict[str, Any]:
    """Severity B: κ ∈ [10, 1000] ATP/bit supported; (1000, 10000] requires
    explanation; κ < 10 or κ > 10000 falsifies the thermodynamic bridge.
    Delegates the banding to energy.thermodynamics.classify_kappa_consistency.
    """
    classification = classify_kappa_consistency(kappa)
    verdict_map: dict[str, Verdict] = {
        "consistent": "confirmed",
        "requires_explanation": "requires_explanation",
        "bridge_inconsistent": "falsified",
    }
    return _result(
        "thermodynamic_bridge",
        verdict_map[classification["classification"]],
        {"kappa": kappa, **classification},
    )


def check_phase_transition_signatures(
    S_history: np.ndarray,
    theta: float,
    baseline_window: int = 50,
    min_samples: int = 100,
) -> dict[str, Any]:
    """Severity B: bistability (Cohen's d > 0.8) + critical slowing (> 20%)
    + elevated susceptibility (> 1.5x) must co-occur. Only 1-of-3 confirmed
    -> revise to a graded-transition model (partial falsification).
    Delegates the underlying measurement to
    analysis.stability.measure_criticality_signatures.
    """
    from analysis.stability import measure_criticality_signatures

    signatures = measure_criticality_signatures(S_history, theta, baseline_window, min_samples)
    if "error" in signatures:
        return _result(
            "phase_transition_signatures",
            "indeterminate",
            {"signatures": signatures},
        )
    n_confirmed = sum(
        [
            signatures["cohens_d_criterion"],
            signatures["susceptibility_criterion"],
            signatures["autocorr_criterion"],
        ]
    )
    verdict: Verdict
    if signatures["all_criteria_met"]:
        verdict = "confirmed"
    elif n_confirmed <= 1:
        verdict = "falsified"
    else:
        verdict = "indeterminate"
    return _result(
        "phase_transition_signatures",
        verdict,
        {"n_signatures_confirmed": n_confirmed, "signatures": signatures},
    )


def check_oscillatory_gate(
    gamma_conscious: np.ndarray,
    gamma_near_miss: np.ndarray,
    n_min: int = 30,
) -> dict[str, Any]:
    """Severity B: inter-level Γ coherence must be significantly higher on
    conscious-access trials than near-miss trials (Cohen's d > 0.5, n >= 30).
    Falsifier: Γ statistically equivalent across two independent datasets
    -> absorb into the PAC precision term.
    Args:
        gamma_conscious: Per-trial Γ^(l) values on conscious-access trials
        gamma_near_miss: Per-trial Γ^(l) values on near-miss trials
        n_min: Minimum trials per condition for a confirmatory test
    """
    gc = np.asarray(gamma_conscious, dtype=float)
    gm = np.asarray(gamma_near_miss, dtype=float)
    n = min(len(gc), len(gm))
    if len(gc) < 2 or len(gm) < 2:
        return _result(
            "oscillatory_gate",
            "indeterminate",
            {"n_conscious": len(gc), "n_near_miss": len(gm), "note": "insufficient trials"},
        )
    pooled_sd = np.sqrt(
        ((len(gc) - 1) * np.var(gc, ddof=1) + (len(gm) - 1) * np.var(gm, ddof=1))
        / (len(gc) + len(gm) - 2)
    )
    cohens_d = float((np.mean(gc) - np.mean(gm)) / (pooled_sd + 1e-12))
    verdict: Verdict
    if n < n_min:
        verdict = "indeterminate"
    elif cohens_d > 0.5:
        verdict = "confirmed"
    else:
        verdict = "falsified"
    return _result(
        "oscillatory_gate",
        verdict,
        {"cohens_d": cohens_d, "n_conscious": len(gc), "n_near_miss": len(gm), "n_min": n_min},
    )


def check_timescale_ordering(
    tau_int: float,
    tau_ign: float,
    tau_theta: float,
    level: int,
) -> dict[str, Any]:
    """Severity B: τ_int,ℓ > τ_ign,ℓ > τ_θ,ℓ for ℓ >= 2 (ordering collapses
    toward equality at ℓ = 1, so level 1 is not evaluated against this rule).
    """
    if level < 2:
        return _result(
            "timescale_ordering",
            "indeterminate",
            {
                "level": level,
                "note": "ordering is only a falsifiable prediction for level >= 2; "
                "it collapses toward equality at level 1",
            },
        )
    ordered = tau_int > tau_ign > tau_theta
    return _result(
        "timescale_ordering",
        "confirmed" if ordered else "falsified",
        {
            "level": level,
            "tau_int": tau_int,
            "tau_ign": tau_ign,
            "tau_theta": tau_theta,
            "ordered": ordered,
        },
    )


def summarize_registry(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize a list of check_* results into a pass/fail/indeterminate
    tally, broken out by severity — the framework-level view implied by §27
    ("a Severity A result may not be reclassified to Severity B after the
    data are seen").
    """
    tally: dict[str, dict[str, int]] = {
        "A": {"confirmed": 0, "falsified": 0, "requires_explanation": 0, "indeterminate": 0},
        "B": {"confirmed": 0, "falsified": 0, "requires_explanation": 0, "indeterminate": 0},
        "C": {"confirmed": 0, "falsified": 0, "requires_explanation": 0, "indeterminate": 0},
    }
    for r in results:
        tally[r["severity"]][r["verdict"]] += 1
    any_severity_a_falsified = tally["A"]["falsified"] > 0
    return {
        "tally": tally,
        "any_severity_a_falsified": any_severity_a_falsified,
        "results": results,
    }
