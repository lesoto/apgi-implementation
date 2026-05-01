# Hierarchical Multi-Timescale APGI System Guide

Enable the full hierarchical system with a single configuration parameter:

```python
from pipeline import APGIPipeline

# Simple: enable full hierarchical system
config = {
    'hierarchical_mode': 'full',
    'n_levels': 4,
    'tau_0': 10.0,  # Base timescale (ms)
    'k': 1.6,       # Timescale ratio
}

pipeline = APGIPipeline(config)
```

---

## Hierarchical Modes

The `hierarchical_mode` parameter simplifies configuration by
consolidating three separate flags:

### Mode: `'off'` (Default)

Disables all hierarchical features. Single-scale APGI system.

```python
config = {'hierarchical_mode': 'off'}
# Equivalent to:
# use_hierarchical = False
# use_hierarchical_precision_ode = False
# use_phase_modulation = False
```

### Mode: `'basic'`

Enables hierarchical multi-timescale processing without advanced features.

```python
config = {'hierarchical_mode': 'basic'}
# Equivalent to:
# use_hierarchical = True
# use_hierarchical_precision_ode = False
# use_phase_modulation = False
```

**Features:**

- Per-level error computation at different timescales
- Multi-scale signal aggregation
- Independent precision at each level

### Mode: `'advanced'`

Enables hierarchical system with precision ODE coupling.

```python
config = {'hierarchical_mode': 'advanced'}
# Equivalent to:
# use_hierarchical = True
# use_hierarchical_precision_ode = True
# use_phase_modulation = False
```

**Features:**

- All 'basic' features
- Precision coupling ODE:

  ```text
  dΠ_ℓ/dt = -Π_ℓ/τ_Π + α|ε_ℓ| + C_down(Π_{ℓ+1} - Π_ℓ) + C_up·ψ(ε_{ℓ-1})
  ```

- Top-down and bottom-up precision coupling

### Mode: `'full'` (Recommended)

Enables all hierarchical features including phase-amplitude coupling.

```python
config = {'hierarchical_mode': 'full'}
# Equivalent to:
# use_hierarchical = True
# use_hierarchical_precision_ode = True
# use_phase_modulation = True
```

**Features:**

- All 'advanced' features
- Phase-amplitude coupling (PAC):
  `θ_ℓ = θ_{0,ℓ}·[1 + κ_down·Π_{ℓ+1}·cos(φ_{ℓ+1})]`
- Bottom-up threshold cascade
- Oscillatory phase dynamics at each level

---

## Architecture Overview

The hierarchical system implements multi-scale error processing across L levels:

```text
Level 0 (Fast):    τ_0 = 10ms    → z_e^(0), z_i^(0)    → Π_0, θ_0
Level 1:           τ_1 = 16ms    → z_e^(1), z_i^(1)    → Π_1, θ_1
Level 2:           τ_2 = 26ms    → z_e^(2), z_i^(2)    → Π_2, θ_2
Level 3 (Slow):    τ_3 = 42ms    → z_e^(3), z_i^(3)    → Π_3, θ_3
                                        ↓
                            Multi-scale aggregation
                                        ↓
                            S = Σ_ℓ w_ℓ·Π_ℓ·|Φ_ℓ|
```

### Timescale Hierarchy

Timescales follow a geometric progression:

```text
τ_ℓ = τ_0 · k^ℓ
```

**Parameters:**

- `τ_0`: Base timescale (fastest level, e.g., 10ms)
- `k`: Timescale ratio (recommended: 1.3-2.0, default: 1.6)
- `n_levels`: Number of levels (default: 4)

**Example:**

```python
config = {
    'hierarchical_mode': 'full',
    'tau_0': 10.0,  # 10ms
    'k': 1.6,       # Timescale ratio
    'n_levels': 4,  # 4 levels
}

# Resulting timescales:
# Level 0: 10.0 ms
# Level 1: 16.0 ms
# Level 2: 25.6 ms
# Level 3: 40.96 ms
```

---

## Per-Level Error Computation (§7)

Each level processes prediction errors independently at its own timescale:

```python
# Spec §7: Φ_ℓ(t+1) = (1 - 1/τ_ℓ)Φ_ℓ(t) + (1/τ_ℓ)z(t)
```

The pipeline automatically computes per-level z-scores:

```python
output = pipeline.step(x_e, x_i, x_hat_e, x_hat_i)

# Single-scale (hierarchical_mode='off'):
# z_e_norm, z_i_norm are scalars

# Multi-scale (hierarchical_mode='basic' or higher):
# Per-level z-scores are computed internally
# Aggregated into multi-scale signal S
```

**Implementation Details:**

- Each level maintains independent running statistics (μ_ℓ, σ²_ℓ)
- Adaptation rate at level ℓ: `α_ℓ = 1/τ_ℓ` (faster at shorter timescales)
- Vectorized computation for performance

---

## Precision Coupling ODE (§2.5)

When `hierarchical_mode='advanced'` or `'full'`, precision evolves with coupling:

```text
dΠ_ℓ/dt = -Π_ℓ/τ_Π + α|ε_ℓ| + C_down(Π_{ℓ+1} - Π_ℓ) + C_up·ψ(ε_{ℓ-1})
```

**Components:**

- `-Π_ℓ/τ_Π`: Self-decay of precision
- `α|ε_ℓ|`: Error-driven precision gain
- `C_down(Π_{ℓ+1} - Π_ℓ)`: Top-down coupling from higher level
- `C_up·ψ(ε_{ℓ-1})`: Bottom-up coupling from lower level error

**Configuration:**

```python
config = {
    'hierarchical_mode': 'advanced',
    'tau_pi': 1000.0,      # Precision decay timescale (ms)
    'alpha_pi': 0.1,       # Error-to-precision gain
    'C_down': 0.1,         # Top-down coupling strength
    'C_up': 0.05,          # Bottom-up coupling strength
}
```

---

## Phase-Amplitude Coupling (§8.4)

When `hierarchical_mode='full'`, higher levels modulate lower
thresholds via oscillatory phase:

```text
θ_ℓ = θ_{0,ℓ}·[1 + κ_down·Π_{ℓ+1}·cos(φ_{ℓ+1})]
```

This creates rhythmic windows of opportunity for ignition at lower levels.

**Configuration:**

```python
config = {
    'hierarchical_mode': 'full',
    'kappa_down': 0.1,     # Phase coupling strength
    'kappa_up': 0.05,      # Bottom-up cascade strength
}
```

**Effect:**

- When `cos(φ_{ℓ+1}) > 0`: Threshold is lowered (easier to ignite)
- When `cos(φ_{ℓ+1}) < 0`: Threshold is raised (harder to ignite)
- Creates phase-locked ignition windows

---

## Bottom-Up Threshold Cascade (§8.4)

Lower-level ignition suppresses higher-level thresholds:

```text
θ_ℓ ← θ_ℓ·[1 - κ_up·H(S_{ℓ-1} - θ_{ℓ-1})]
```

where `H(·)` is the Heaviside function (1 if superthreshold, else 0).

**Effect:**

- When level ℓ-1 ignites: Level ℓ threshold is reduced
- Propagates salience upward through hierarchy
- Implements hierarchical attention

---

## Multi-Scale Signal Aggregation (§8.2)

The total signal aggregates contributions from all levels:

```text
S(t) = Σ_ℓ w_ℓ · Π_ℓ^eff(t) · |Φ_ℓ(t)|
```

**Weights:**

```text
w_ℓ = k^{-ℓ} / Z    (geometrically decreasing)
```

**Example (k=1.6, 4 levels):**

```text
w_0 = 0.4167  (fastest level, highest weight)
w_1 = 0.2604
w_2 = 0.1628
w_3 = 0.1018  (slowest level, lowest weight)
```

---

## Configuration Examples

### Example 1: Basic Hierarchical System

```python
config = {
    'hierarchical_mode': 'basic',
    'n_levels': 3,
    'tau_0': 10.0,
    'k': 1.5,
    'alpha_e': 0.1,
    'alpha_i': 0.1,
    'lambda': 0.1,
    'eta': 0.01,
}

pipeline = APGIPipeline(config)
```

### Example 2: Full Hierarchical System with Precision ODE

```python
config = {
    'hierarchical_mode': 'full',
    'n_levels': 4,
    'tau_0': 10.0,
    'k': 1.6,
    
    # Precision ODE parameters
    'tau_pi': 1000.0,
    'alpha_pi': 0.1,
    'C_down': 0.1,
    'C_up': 0.05,
    
    # Phase-amplitude coupling
    'kappa_down': 0.1,
    'kappa_up': 0.05,
    
    # Signal accumulation
    'lambda': 0.1,
    'eta': 0.01,
}

pipeline = APGIPipeline(config)
```

### Example 3: Hierarchical + Reservoir + Kuramoto

```python
config = {
    'hierarchical_mode': 'full',
    'n_levels': 4,
    'tau_0': 10.0,
    'k': 1.6,
    
    # Reservoir computing
    'use_reservoir': True,
    'reservoir_size': 100,
    'spectral_radius': 0.9,
    
    # Kuramoto oscillators
    'use_kuramoto': True,
    'kuramoto_n_levels': 4,
    'kuramoto_coupling': 0.1,
}

pipeline = APGIPipeline(config)
```

---

## Output Structure

The `step()` function returns a dictionary with hierarchical information:

```python
output = pipeline.step(x_e, x_i, x_hat_e, x_hat_i)

# Core outputs (all modes)
output['z_e']              # Exteroceptive error
output['z_i']              # Interoceptive error
output['S']                # Accumulated signal
output['theta']            # Dynamic threshold
output['ignition_margin']  # Δ(t) = S(t) - θ(t)
output['B']                # Ignition state (0 or 1)

# Hierarchical outputs (hierarchical_mode != 'off')
output['S_hierarchical']   # Multi-scale aggregated signal
output['pi_levels']        # Precision at each level
output['theta_levels']     # Threshold at each level
output['phases']           # Oscillatory phases (if use_kuramoto=True)
```

---

## Validation and Diagnostics

### Check Hierarchical Configuration

```python
from core.validation import validate_config

config = {'hierarchical_mode': 'full', 'n_levels': 4}
validate_config(config)  # Raises ValidationError if invalid
```

### Monitor Hierarchical Dynamics

```python
# Run simulation
for t in range(1000):
    output = pipeline.step(x_e_t, x_i_t)
    
    # Monitor multi-scale signal
    print(f"S = {output['S']:.3f}")
    print(f"θ = {output['theta']:.3f}")
    print(f"Δ = {output['ignition_margin']:.3f}")
    
    # Monitor precision coupling (if enabled)
    if 'pi_levels' in output:
        print(f"Π_levels = {output['pi_levels']}")
```

### Validate Spectral Signature

```python
from stats.spectral_model import validate_spectral_signature

# Collect signal history
S_history = [output['S'] for output in outputs]

# Validate 1/f spectrum
result = validate_spectral_signature(
    np.array(S_history),
    taus=pipeline.taus,
    dt=1.0
)

print(result['message'])
# Output: "Spectral exponent β=1.05 (Hurst H=0.53).
# Healthy range: [0.8, 1.5]. ✅ Valid"
```

---

## Troubleshooting

### Issue: Hierarchical system not activating

**Problem:** Configuration has `hierarchical_mode='full'` but system
behaves like single-scale.

**Solution:** Check that `hierarchical_mode` is set before creating pipeline:

```python
# ✅ Correct
config = {'hierarchical_mode': 'full'}
pipeline = APGIPipeline(config)

# ❌ Wrong (hierarchical_mode not set)
config = {}
pipeline = APGIPipeline(config)
```

### Issue: Precision coupling diverges

**Problem:** Precision values grow unbounded or become NaN.

**Solution:** Reduce coupling strengths or increase decay timescale:

```python
config = {
    'hierarchical_mode': 'advanced',
    'tau_pi': 2000.0,      # Increase decay timescale
    'C_down': 0.05,        # Reduce coupling strength
    'C_up': 0.02,
}
```

### Issue: Threshold oscillates wildly

**Problem:** Threshold exhibits unstable oscillations.

**Solution:** Reduce phase coupling strength or disable phase
modulation:

```python
config = {
    'hierarchical_mode': 'advanced',  # Skip 'full' to disable phase modulation
    'kappa_down': 0.05,               # Reduce phase coupling
}
```

---

## Performance Considerations

### Computational Cost

- `'off'`: 1.0x - Single-scale baseline
- `'basic'`: 1.5x - Per-level error computation
- `'advanced'`: 2.0x - + Precision ODE integration
- `'full'`: 2.5x - + Phase dynamics

### Memory Usage

```text
Single-scale:  ~50 MB
4-level hierarchical: ~80 MB
4-level + reservoir: ~150 MB
4-level + Kuramoto: ~120 MB
```

### Optimization Tips

1. **Reduce number of levels** if performance is critical
2. **Use 'basic' mode** instead of 'full' if phase coupling not needed
3. **Disable reservoir** if not using reservoir computing
4. **Vectorize error computation** (already done in pipeline)

---

## References

- **Spec §7:** Hierarchical Multi-Timescale Architecture
- **Spec §8:** Oscillatory Phase Coupling
- **Spec §2.5:** Precision ODE
- **Spec §14:** Observable Mapping

---

## Examples

See `examples/06_hierarchical_system.py` for complete working example.

---

**Last Updated:** April 21, 2026  
**Status:** Production Ready ✅
