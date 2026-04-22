# Hierarchical Multi-Timescale APGI

## Quick Start

Enable full hierarchical system:

```python
config = {'hierarchical_mode': 'full'}
```

## Architecture

The hierarchical system implements multi-scale error processing:

- Level 0: Fast timescale τ_0 (e.g., 10ms)
- Level 1: Medium timescale τ_1 = τ_0 · k (e.g., 16ms)
- Level 2: Slow timescale τ_2 = τ_0 · k² (e.g., 26ms)
- ...

Each level processes prediction errors independently, then aggregates.

## Configuration

```python
config = {
    'hierarchical_mode': 'full',
    'n_levels': 4,
    'tau_0': 10.0,  # Base timescale (ms)
    'k': 1.6,       # Timescale ratio
    'use_hierarchical_precision_ode': True,
    'use_phase_modulation': True,
}
```

## Hierarchical Mode Presets

The `hierarchical_mode` parameter provides a simple way to enable hierarchical features:

| Mode | Features Enabled |
| ------ | ------------------ |
| `'off'` | No hierarchical features (default) |
| `'basic'` | Hierarchical multiscale integration only |
| `'advanced'` | + Precision coupling ODE |
| `'full'` | + Phase-amplitude coupling |

### Mode Details

**Off Mode:**

```python
config = {'hierarchical_mode': 'off'}  # Default
# Equivalent to:
# use_hierarchical = False
# use_hierarchical_precision_ode = False
# use_phase_modulation = False
```

**Basic Mode:**

```python
config = {'hierarchical_mode': 'basic'}
# Enables multi-scale integration with per-level error processing
# Equivalent to:
# use_hierarchical = True
# use_hierarchical_precision_ode = False
# use_phase_modulation = False
```

**Advanced Mode:**

```python
config = {'hierarchical_mode': 'advanced'}
# Adds precision coupling ODE for dynamic precision evolution
# Equivalent to:
# use_hierarchical = True
# use_hierarchical_precision_ode = True
# use_phase_modulation = False
```

**Full Mode:**

```python
config = {'hierarchical_mode': 'full'}
# All hierarchical features including phase-amplitude coupling
# Equivalent to:
# use_hierarchical = True
# use_hierarchical_precision_ode = True
# use_phase_modulation = True
```

## Features

### 1. Per-Level Error Processing

Each level computes z-scores at its own timescale using per-level EMA statistics:

```python
# For each level ℓ:
mu_e_ℓ(t+1) = (1 - α_ℓ)·μ_e_ℓ(t) + α_ℓ·ε_e(t)
sigma2_e_ℓ(t+1) = (1 - α_ℓ)·sigma2_e_ℓ(t) + α_ℓ·(ε_e(t) - μ_e_ℓ(t))²
z_e_ℓ(t) = (ε_e(t) - μ_e_ℓ(t)) / sqrt(sigma2_e_ℓ(t) + ε)
```

Where `α_ℓ = 1/τ_ℓ` for level-specific adaptation rates.

### 2. Precision Coupling ODE

Precision evolves with top-down and bottom-up coupling:

```text
dΠ_ℓ/dt = -Π_ℓ/τ_Π + α|ε_ℓ| + C_down(Π_{ℓ+1} - Π_ℓ) + C_up·ψ(ε_{ℓ-1})
```

Parameters:

- `tau_pi`: Precision timescale (default: 1000.0 ms)
- `C_down`: Top-down coupling strength (default: 0.1)
- `C_up`: Bottom-up coupling strength (default: 0.05)
- `alpha_gain`: Error-to-precision gain (default: 0.1)

### 3. Phase-Amplitude Coupling

Higher levels modulate lower thresholds via oscillatory phase:

```text
θ_ℓ = θ_{0,ℓ}·[1 + κ_down·Π_{ℓ+1}·cos(φ_{ℓ+1})]
```

Parameters:

- `kappa_phase`: Phase modulation strength (default: 0.1)
- `omega_phases`: Oscillator frequencies per level (default: [0.1, 0.05, 0.01])

### 4. Bottom-Up Cascade

Lower-level ignition suppresses higher thresholds:

```text
θ_ℓ ← θ_ℓ·[1 - κ_up·H(S_{ℓ-1} - θ_{ℓ-1})]
```

Where `H()` is the Heaviside step function.

## Complete Configuration Example

```python
from pipeline import APGIPipeline

config = {
    # Base configuration
    'alpha_e': 0.1,
    'alpha_i': 0.1,
    'lambda': 0.1,
    'eta': 0.01,
    'delta': 0.5,
    'kappa': 0.1,
    
    # Hierarchical system
    'hierarchical_mode': 'full',
    'n_levels': 4,
    'tau_0': 10.0,
    'k': 1.6,
    
    # Precision ODE parameters
    'tau_pi': 1000.0,
    'C_down': 0.1,
    'C_up': 0.05,
    'alpha_gain': 0.1,
    
    # Phase modulation
    'kappa_phase': 0.1,
    'omega_phases': [0.1, 0.05, 0.01, 0.005],
}

pipeline = APGIPipeline(config)

# Run simulation
for t in range(1000):
    result = pipeline.step(x_e[t], x_i[t])
    # Access per-level precision values
    if 'hierarchical_pis' in result:
        print(f"Level precisions: {result['hierarchical_pis']}")
```

## Timescale Hierarchy

The system builds timescales geometrically:

```python
from hierarchy.multiscale import build_timescales

taus = build_timescales(tau0=10.0, k=1.6, n_levels=4)
# Result: [10.0, 16.0, 25.6, 40.96] ms
```

Recommended parameter ranges:

- `tau_0`: 5-50 ms (fastest processing level)
- `k`: 1.3-2.0 (timescale ratio, per spec §8.1)
- `n_levels`: 3-6 (more levels = slower processing)

## Multi-Scale Signal Aggregation

The hierarchical system aggregates signals across levels:

```text
S(t) = Σ_ℓ w_ℓ · Π_ℓ^eff · |Φ_ℓ(t)|
```

Where:

- `w_ℓ = k^{-ℓ} / Z` are geometrically decreasing weights
- `Φ_ℓ(t)` are multi-scale features updated at each level's timescale
- `Π_ℓ^eff` are level-specific effective precisions

## Validation

The hierarchical system produces 1/f (pink noise) spectral characteristics in threshold dynamics:

```python
# Run long simulation for spectral analysis
pipeline = APGIPipeline(config)
for t in range(10000):
    pipeline.step(x_e[t], x_i[t])

# Validate 1/f spectrum
from pipeline import validate_pink_noise
results = pipeline.validate()
print(f"Spectral exponent: {results['beta']:.2f}")
print(f"Hurst exponent: {results['hurst']:.2f}")
```

Expected values:

- Spectral exponent β ≈ 1.0 (pink noise)
- Hurst exponent H ≈ 0.7-0.9 (long-range correlations)

## Performance Considerations

- Each additional level adds computation overhead
- Per-level error processing uses O(n_levels) memory
- Precision ODE adds minimal overhead per step
- Phase modulation requires phase tracking per level

## Troubleshooting

**Issue:** Hierarchical system not enabled

- Check: `config['hierarchical_mode']` is set correctly
- Verify: `pipeline.use_hierarchical` is True after initialization

**Issue:** Timescales too fast/slow

- Adjust `tau_0` for base timescale
- Adjust `k` for spacing between levels

**Issue:** Numerical instability

- Reduce `C_down` or `C_up` coupling strengths
- Increase `tau_pi` for slower precision dynamics
- Check that `dt` satisfies stability condition: `dt ≤ min(τ) / 10`

## References

- Spec §7: Hierarchical Multi-Timescale Integration
- Spec §8: Cross-Scale Coupling
- Spec §9: Oscillatory Synchronization
- `examples/06_hierarchical_system.py` - Complete working example
