# APGI Implementation Validation Report

## Scope
This report validates the Python implementation in this repository against the "APGI Unified Mathematical Specification" across:

1. signal preprocessing
2. precision dynamics
3. ignition
4. allostatic threshold
5. reservoir implementation
6. hierarchical coupling
7. statistical validation

## Overall Rating

**64 / 100**

## Rating Rationale

### What is implemented well
- Core prediction error flow exists (`epsilon = x - x_hat`) and is used in the pipeline.  
- Precision from inverse variance with clamping is implemented and wired.  
- Dopamine as additive interoceptive error bias is implemented (`z_i_eff = z_i + beta`).  
- Soft ignition (`sigmoid`) and hard ignition (`S > theta`) are both available with stochastic Bernoulli sampling.  
- Threshold decay toward baseline and refractory increment mechanisms are implemented.  
- Hierarchical precision coupling ODE and phase coupling scaffolding exist.  
- Spectral/Hurst utilities exist and are integrated into pipeline validation.

### Main correctness and spec-alignment gaps
- `signal_drift` adds dopamine directly to drift (`+ beta`) rather than using `|z_i_eff|`; this diverges from the canonical signal equation unless explicitly using the alternative mode with matching semantics.  
- `core/allostatic.py::allostatic_threshold_ode` omits the `eta * (C - V)` term despite claiming the full equation in docstring (the pipeline adds it externally, but the function itself is incomplete relative to its own stated formula).  
- Two threshold ODE implementations exist with conflicting formulations; `core/threshold.py::update_threshold_ode` still uses derivative coupling with `dS/dt`, which the spec explicitly removes.  
- `hierarchy/coupling.py::estimate_hierarchy_levels` uses `overlap_factor` naming and computes with `ceil(log10 ratio / log10 overlap_factor)`; spec defines exact formula with `k` and `floor(...) + 1`.  
- `LiquidNetwork.readout_signal` uses `x^T x` while the spec recommends trained linear readout `W_out^T x`.  
- Runtime functionality cannot be fully validated in this environment because required dependencies (at least `numpy`) are not installed.

## Section-by-section scorecard

| Area | Score | Notes |
|---|---:|---|
| Signal preprocessing | 78 | EMA/windowed stats and z-scoring are present. |
| Precision dynamics | 80 | Inverse variance, clamping, ACh/NE/DA forms mostly aligned. |
| Ignition | 90 | Soft/hard decision + margin are implemented correctly. |
| Allostatic threshold | 58 | Core components present, but duplicate/conflicting ODE APIs reduce correctness clarity. |
| Reservoir implementation | 52 | Dynamics present, but readout differs from recommended linear trained output. |
| Hierarchical coupling | 60 | Core couplings exist; hierarchy level estimation formula differs from spec canonical form. |
| Statistical validation | 70 | PSD/Hurst tooling is present and connected; practical execution unverified due env deps. |

## Recommended high-impact fixes
1. Unify signal equation path so dopamine treatment is internally consistent with chosen mode (prefer canonical `z_i_eff = z_i + beta` then `|z_i_eff|`).
2. Remove or clearly deprecate threshold APIs that still use derivative coupling to `dS/dt`.
3. Align hierarchy level-count helper exactly to `L = floor(log(tau_max/tau_min)/log(k)) + 1`.
4. Add trained `W_out` linear readout path to `LiquidNetwork` and make it the default APGI-compatible readout.
5. Add automated tests validating each canonical equation and one end-to-end timestep against known reference values.

## Validation constraints
- End-to-end runtime validation was blocked by missing Python dependencies in this environment (`ModuleNotFoundError: numpy`).
