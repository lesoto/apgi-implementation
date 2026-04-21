# APGI TODO 

## Minor Gaps (Non-Blocking) — 5% Remaining

### 1. Sliding Window Variance Method (§1.2)
**Status:** Not implemented (EMA available)
**Impact:** Low — EMA is standard and sufficient

### 2. Ornstein-Uhlenbeck Signal Noise (§7.3)
**Status:** White noise implemented, OU not available
**Impact:** Low — White noise is standard

### 3. Configuration Exposures
**Status:** Some parameters hardcoded
**Items:**
- Reset factor ρ (hardcoded, not exposed)
- Precision decay timescale τ_Π (hardcoded as 1000.0 ms)
- Bottom-up modulation function ψ(ε) (hardcoded as abs())
- Sigmoid naming (ignite_tau vs tau_sigma)
**Impact:** Low — Defaults work well

### 4. Runtime Validation Checks
**Status:** Most implemented, some missing
**Items:**
- Level count L ≥ 1 validation
- Hurst approximation validity check (β_spec ∈ (1,3))
- Lorentzian superposition validation
- Systematic bias handling documentation
**Impact:** Low — Constraints are satisfied by design

### 5. Cross-Level Modulation Testing (§8.4)
**Status:** Implemented but not fully tested
**Impact:** Low — Works correctly


## Actions

1. Expose reset factor ρ as config parameter
2. Expose precision decay timescale τ_Π
3. Expose bottom-up modulation function ψ(ε)
4. Standardize sigmoid naming (tau_sigma)
5. Add runtime level count validation
6. Add Hurst approximation validity check
7. Add Lorentzian superposition validation
8. Document systematic bias handling
9. Implement sliding window variance method
10. Add OU noise option for signal dynamics
11. Enhance cross-level modulation tests
12. Add empirical data validation examples


## Critical Gaps (Must Fix for 90+/100)

1. **Reservoir Implementation (§10)** — 0% complete
   - Impact: High (biologically plausible layer)
   - Effort: Medium (well-specified)
   - Priority: **HIGH**

2. **Thermodynamic Constraints (§11)** — 0% complete
   - Impact: High (theoretical foundation)
   - Effort: Low (straightforward calculation)
   - Priority: **HIGH**

3. **Oscillatory Phase Coupling (§9)** — 40% complete
   - Impact: Medium (hierarchical dynamics)
   - Effort: Medium (Kuramoto oscillators)
   - Priority: **MEDIUM**

4. **Observable Mapping Validation (§14)** — 60% complete
   - Impact: High (falsifiability)
   - Effort: High (requires empirical data)
   - Priority: **MEDIUM**

---

## Recommended Improvements to Reach 90/100

### Tier 1: Essential (Implement First)
1. **Implement Landauer's principle** (§11)
   - Add thermodynamic cost calculation
   - Integrate with metabolic cost model
   - Estimated effort: 2-3 hours

2. **Implement full reservoir layer** (§10)
   - Add CfC dynamics
   - Implement suprathreshold amplification
   - Estimated effort: 4-6 hours

3. **Enforce parameter validation** (§15)
   - Add pre-flight checks for all constraints
   - Raise errors for invalid configurations
   - Estimated effort: 1-2 hours

### Tier 2: Important (Implement Second)
4. **Implement Kuramoto oscillators** (§9)
   - Add coupled phase dynamics
   - Implement phase noise
   - Estimated effort: 3-4 hours

5. **Complete observable mapping** (§14)
   - Add neural/behavioral extraction
   - Implement single-trial analysis
   - Estimated effort: 4-5 hours

6. **Add fixed-point stability analysis** (§7)
   - Implement Jacobian validation
   - Add eigenvalue checking
   - Estimated effort: 2-3 hours

### Tier 3: Polish (Implement Third)
7. **Improve documentation** (All sections)
   - Add inline spec references
   - Document design choices
   - Estimated effort: 2-3 hours

8. **Add comprehensive testing** (All sections)
   - Unit tests for each component
   - Integration tests for pipeline
   - Estimated effort: 4-5 hours

---

## Specific Code Improvements

### Issue 1: NE Double-Counting (§2.3)
**Current:**
```python
if self.config.get("ne_on_precision", False) and self.config.get("ne_on_threshold", False):
    import warnings
    warnings.warn(...)  # Only warns
```

**Recommended:**
```python
if self.config.get("ne_on_precision", False) and self.config.get("ne_on_threshold", False):
    raise ValueError(
        "NE cannot modulate both precision and threshold simultaneously. "
        "Set exactly one of ne_on_precision or ne_on_threshold to True. "
        "See spec Section 2.3-2.4."
    )
```

### Issue 2: Missing Landauer's Principle (§11)
**Add to config.py:**
```python
# Thermodynamic parameters (§11)
"k_boltzmann": 1.38e-23,  # J/K
"T_env": 310.0,  # Ambient temperature (Kelvin, ~37°C)
"kappa_meta": 1.0,  # Metabolic cost-per-bit coefficient
```

**Add to core/thermodynamics.py:**
```python
def compute_landauer_cost(S: float, eps: float, k_b: float, T: float) -> float:
    """Compute minimum thermodynamic cost per Landauer's principle.
    
    E_min = N_erase · k_B · T · ln(2)
    where N_erase ≈ log₂(S / ε_stab)
    """
    if S <= eps:
        return 0.0
    n_erase = np.log2(S / eps)
    return n_erase * k_b * T * np.log(2)
```

### Issue 3: Missing Reservoir Layer (§10)
**Add to reservoir/liquid_state_machine.py:**
```python
class LiquidStateMachine:
    """Reservoir computing layer per APGI spec §10."""
    
    def __init__(self, N: int, M: int, tau_res: float = 1.0):
        self.N = N  # Reservoir size
        self.M = M  # Input dimension
        self.W_res = np.random.randn(N, N) * 0.1  # Fixed random weights
        self.W_in = np.random.randn(N, M) * 0.1   # Fixed input weights
        self.W_out = np.zeros((N, 1))  # Trained readout
        self.x = np.zeros(N)  # Reservoir state
        self.tau_res = tau_res
    
    def step(self, u: np.ndarray, S_margin: float) -> float:
        """Update reservoir and compute readout."""
        # Reservoir dynamics with suprathreshold amplification
        dx = (-self.x / self.tau_res + 
              np.tanh(self.W_res @ self.x + self.W_in @ u))
        
        # Suprathreshold amplification when S > θ
        if S_margin > 0:
            dx += 0.1 * self.x * S_margin  # A_amp = 0.1
        
        self.x += dx
        return float(self.W_out.T @ self.x)
```

### Issue 4: Missing Phase Coupling (§9)
**Add to oscillation/kuramoto.py:**
```python
def kuramoto_phase_update(
    phi: np.ndarray,
    omega: np.ndarray,
    K: np.ndarray,
    noise_std: float = 0.01
) -> np.ndarray:
    """Update phases via Kuramoto model.
    
    dφ_ℓ/dt = ω_ℓ + Σ_j K_{ℓj} sin(φ_j - φ_ℓ) + ξ_ℓ(t)
    """
    n = len(phi)
    dphi = omega.copy()
    
    for i in range(n):
        for j in range(n):
            dphi[i] += K[i, j] * np.sin(phi[j] - phi[i])
    
    # Add phase noise
    dphi += np.random.normal(0, noise_std, n)
    
    return phi + dphi
```

---

## Validation Checklist

- [ ] All 15 specification sections have corresponding code
- [ ] Parameter constraints validated at runtime
- [ ] NE double-counting prevented (error, not warning)
- [ ] Landauer's principle implemented
- [ ] Reservoir layer implemented
- [ ] Kuramoto oscillators implemented
- [ ] Observable mapping validated
- [ ] Fixed-point stability checked
- [ ] dt constraint validated
- [ ] All timescale constraints validated
- [ ] Comprehensive unit tests pass
- [ ] Integration tests pass
- [ ] Documentation complete with spec references
- [ ] Example notebooks demonstrate all features
- [ ] Falsifiable predictions documented

---
