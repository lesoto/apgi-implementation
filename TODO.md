# APGI TODO — Implementation Evaluation Update (2026-04-21)

This file was updated after a static code audit of the current repository against the **APGI Unified Mathematical Specification** pipeline:

**Signal preprocessing → precision dynamics → ignition → allostatic threshold → reservoir implementation → hierarchical coupling → statistical validation**.

> Note: this assessment is based on code inspection and available tests. Full runtime verification is currently blocked in this environment because `numpy` is not installed for `pytest` collection.

---

## Overall Ratings (1–100)

- **Correctness:** **72/100**
- **Accuracy to specification:** **68/100**
- **Completeness:** **77/100**

### Why these scores

- Strong coverage exists for most major subsystems (precision, threshold, Kuramoto, reservoir, thermodynamics, spectral utilities).
- However, there are several **spec-critical mismatches** in the end-to-end pipeline ordering and equations (notably signal update/reset, β usage in dynamics, and threshold/value wiring differences), which lower correctness/accuracy.
- Completeness is relatively high because nearly all major sections have code artifacts, but some are partial or not fully integrated.

---

## Section-by-Section Audit

## 1) Signal preprocessing

### Status
- Implemented: raw prediction error, EMA mean/variance, z-score-like normalization.
- Optional sliding-window stats exist via `RunningStats`, but are **not integrated as a selectable pipeline method**.

### Findings
- ✅ Raw prediction error matches spec intent.
- ✅ Centered EMA variance is implemented (good handling of bias).
- ⚠️ Sliding-window variance is utility-level only, not a first-class mode with `T_win`/Bessel options.

### Score: **80/100**

---

## 2) Precision dynamics

### Status
- Precision inversion/clamping implemented.
- ACh/NE gains and dopamine additive bias implemented.
- Hierarchical precision ODE implemented in core/hierarchy functions.

### Findings
- ✅ `Π = 1/(σ²+ε)` with clamp is present.
- ✅ DA is additive on interoceptive error (`z_i + β`).
- ⚠️ Pipeline still warns (instead of hard-failing) on NE double-counting in one path.
- ⚠️ Hierarchical ODE integration in pipeline is simplified and not clearly level-generalized from actual per-level errors.

### Score: **75/100**

---

## 3) Ignition mechanism

### Status
- Hard and soft ignition are implemented.
- Sigmoid temperature (`ignite_tau`) and Bernoulli sampling are present.

### Findings
- ✅ Correct hard-threshold convention (`S > θ`).
- ✅ Soft ignition probability and sampling are present.
- ✅ Margin utility exists.

### Score: **88/100**

---

## 4) Allostatic threshold dynamics

### Status
- Core update and decay are implemented.
- Metabolic and information terms are implemented.

### Findings
- ✅ `θ += η(C−V)+δ·B_prev` then decay exists.
- ✅ NE threshold modulation function exists.
- ⚠️ In integrated pipeline, some sequencing/term choices differ from canonical table semantics (especially combined with signal update/reset behavior).

### Score: **72/100**

---

## 5) Reservoir implementation

### Status
- Substantial implementation exists: random fixed reservoir, spectral radius control, step dynamics, readout, ridge training.

### Findings
- ✅ Reservoir state ODE-style update is present.
- ✅ Suprathreshold amplification term is present.
- ✅ Readout APIs included.
- ⚠️ Optional layer integration in pipeline is currently “add-on” rather than a fully alternative execution path replacing steps 7–8 in a spec-explicit mode.

### Score: **85/100**

---

## 6) Hierarchical coupling

### Status
- Helper modules implement multi-timescale integrators, coupling, phase-threshold mechanisms.

### Findings
- ✅ Timescale hierarchy and weighted aggregation utilities exist.
- ✅ Top-down/bottom-up threshold modulation functions exist.
- ⚠️ End-to-end hierarchical execution in `APGIPipeline.step()` is only partial; coupling uses approximated placeholders for some level errors.

### Score: **70/100**

---

## 7) Statistical validation

### Status
- Spectral/Lorentzian and Hurst-related utilities exist.
- Stability analysis module exists.
- Observable mapping module exists.

### Findings
- ✅ Lorentzian superposition + 1/f exponent estimation implemented.
- ✅ Hurst estimation + validation paths present.
- ✅ Stability Jacobian/eigenvalue checks implemented.
- ⚠️ Empirical/behavioral validation is still simulation/proxy-centric, not a full dataset-driven validation pipeline.

### Score: **82/100**

---

## Critical Mismatches to Fix First (highest impact on correctness)

1. **Post-ignition signal reset missing in main pipeline**
   - Spec requires `S ← ρ·S` on ignition.
   - `pipeline.py` does not apply reset factor after `B_t == 1`.

2. **Signal accumulation equation mismatch in integrated path**
   - Spec discrete core uses `S(t+1)=(1−λ)S+λS_inst`.
   - Pipeline currently advances `S` through ODE utility in a way that mixes terms (including β use in dynamics input) rather than directly applying canonical leaky update in the minimal loop.

3. **Config/notation inconsistency**
   - Uses `beta` and `ignite_tau`; spec glossary prefers `β_DA`, `τ_σ`.
   - Not wrong computationally, but increases audit ambiguity and implementation drift risk.

4. **Validation policy inconsistency for NE double-counting**
   - Validator raises error, but pipeline catches validation errors and converts to warning; this can allow forbidden configurations to continue.

5. **Sliding-window method not fully wired**
   - Utility exists, but there is no clean `EMA vs T_win` runtime switch in pipeline flow.

---

## Updated Action Plan

## Tier 1 (must do for 85+ correctness)
1. Implement explicit **post-ignition `S ← ρ·S`** in `APGIPipeline.step()` with validated `reset_factor`.
2. Add a strict **minimal canonical mode** that follows Section 13 step order exactly (including discrete leaky accumulation path).
3. Enforce NE separation as a hard failure at runtime (no warning fallback for invalid dual modulation).
4. Rename/alias config parameters to spec names (`beta_da`, `tau_sigma`) while maintaining backward compatibility.

## Tier 2 (must do for 90+ accuracy)
5. Add true **EMA vs sliding-window switch** with Bessel correction option for small windows.
6. Tighten hierarchical integration by computing per-level errors/states instead of placeholders.
7. Add regression tests for exact timestep ordering and equation fidelity against reference equations.

## Tier 3 (polish/completeness)
8. Add end-to-end validation notebook/script for spectral + observable predictions on synthetic/empirical data.
9. Harmonize docs with actual file names/entry points and remove stale claims.

---

## Validation Run Notes (current environment)

- `pytest -q` currently fails during collection because dependency `numpy` is missing in the runtime environment.
- No behavior-level runtime score adjustment was possible from tests in this environment.

