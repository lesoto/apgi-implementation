# APGI Parameter Constraints Guide

This guide documents all APGI parameters, their valid ranges, physical meanings, and typical values. All parameters are validated at pipeline initialization.

---

## Signal Preprocessing (§1)

### `tau_s` - Signal Timescale

**Type:** Float  
**Valid Range:** (0, ∞)  
**Typical Values:** 1-10 ms  
**Default:** 5.0 ms  
**Physical Meaning:** Time constant for signal accumulation  
**Spec Reference:** §1, §7.1

**Constraints:**

- Must be positive

- Should be ≤ tau_theta and tau_pi

- Affects integration speed

**Guidance:**

- Smaller values: Faster response, more noise

- Larger values: Slower response, more filtering

- Typical: 5-10 ms for cortical timescales

**Example:**

```python
config["tau_s"] = 5.0  # 5 ms signal timescale
```

---

### `tau_pi` - Precision Timescale

**Type:** Float  
**Valid Range:** (0, ∞)  
**Typical Values:** 100-2000 ms  
**Default:** 1000.0 ms  
**Physical Meaning:** Time constant for precision adaptation  
**Spec Reference:** §2.4, §7.3

**Constraints:**

- Must be positive

- Should be ≥ tau_s

- Affects precision learning speed

**Guidance:**

- Smaller values: Faster precision adaptation

- Larger values: Slower precision adaptation

- Typical: 1000 ms for slow learning

**Example:**

```python
config["tau_pi"] = 1000.0  # 1 second precision timescale
```

---

### `tau_theta` - Threshold Timescale

**Type:** Float  
**Valid Range:** (0, ∞)  
**Typical Values:** 100-2000 ms  
**Default:** 1000.0 ms  
**Physical Meaning:** Time constant for threshold adaptation  
**Spec Reference:** §4, §7.3

**Constraints:**

- Must be positive

- Should be ≥ tau_s

- Affects threshold learning speed

**Guidance:**

- Smaller values: Faster threshold adaptation

- Larger values: Slower threshold adaptation

- Typical: 1000 ms for slow learning

**Example:**

```python
config["tau_theta"] = 1000.0  # 1 second threshold timescale
```

---

### `eps_stab` - Stability Constant

**Type:** Float  
**Valid Range:** (0, 1e-6)  
**Typical Values:** 1e-8 to 1e-6  
**Default:** 1e-8  
**Physical Meaning:** Prevents division by zero  
**Spec Reference:** §1.3, §2.2

**Constraints:**

- Must be positive

- Should be very small

- Affects numerical stability

**Guidance:**

- Smaller values: More accurate but less stable

- Larger values: More stable but less accurate

- Typical: 1e-8 for double precision

**Example:**

```python
config["eps_stab"] = 1e-8  # 1e-8 stability constant
```

---

## Precision System (§2)

### `pi_min` - Minimum Precision

**Type:** Float  
**Valid Range:** (0, pi_max)  
**Typical Values:** 0.001-0.1  
**Default:** 0.01  
**Physical Meaning:** Minimum precision (maximum uncertainty)  
**Spec Reference:** §2.2

**Constraints:**

- Must be positive

- Must be < pi_max

- Affects precision range

**Guidance:**

- Smaller values: Allow more uncertainty

- Larger values: Enforce minimum confidence

- Typical: 0.01 for moderate uncertainty

**Example:**

```python
config["pi_min"] = 0.01  # Minimum precision 0.01
```

---

### `pi_max` - Maximum Precision

**Type:** Float  
**Valid Range:** (pi_min, ∞)  
**Typical Values:** 10-1000  
**Default:** 100.0  
**Physical Meaning:** Maximum precision (minimum uncertainty)  
**Spec Reference:** §2.2

**Constraints:**

- Must be > pi_min

- Affects precision range

- Prevents precision explosion

**Guidance:**

- Smaller values: Limit maximum confidence

- Larger values: Allow high confidence

- Typical: 100 for moderate confidence

**Example:**

```python
config["pi_max"] = 100.0  # Maximum precision 100
```

---

### `ne_on_precision` - NE Modulates Precision

**Type:** Boolean  
**Valid Values:** True, False  
**Default:** False  
**Physical Meaning:** Whether NE modulates precision  
**Spec Reference:** §2.3, §15.1

**Constraints:**

- Cannot be True if ne_on_threshold is True

- Enforced at initialization

**Guidance:**

- True: NE increases precision (attention)

- False: NE does not affect precision

- Typical: False (NE modulates threshold instead)

**Example:**

```python
config["ne_on_precision"] = False  # NE does not modulate precision
```

---

### `ne_on_threshold` - NE Modulates Threshold

**Type:** Boolean  
**Valid Values:** True, False  
**Default:** True  
**Physical Meaning:** Whether NE modulates threshold  
**Spec Reference:** §2.3, §15.1

**Constraints:**

- Cannot be True if ne_on_precision is True

- Enforced at initialization

**Guidance:**

- True: NE lowers threshold (arousal)

- False: NE does not affect threshold

- Typical: True (NE modulates threshold)

**Example:**

```python
config["ne_on_threshold"] = True  # NE modulates threshold
```

---

## Signal Accumulation (§3)

### `lam` - Integration Rate

**Type:** Float  
**Valid Range:** (0, 1)  
**Typical Values:** 0.1-0.5  
**Default:** 0.2  
**Physical Meaning:** Leaky integrator rate  
**Spec Reference:** §3.2

**Constraints:**

- Must be > 0

- Must be < 1

- Enforced at initialization

**Guidance:**

- Smaller values: More filtering, slower response

- Larger values: Less filtering, faster response

- Typical: 0.2 for moderate integration

**Example:**

```python
config["lam"] = 0.2  # 20% integration rate
```

---

### `use_log_compression` - Log-Compress Signal

**Type:** Boolean  
**Valid Values:** True, False  
**Default:** False  
**Physical Meaning:** Whether to apply log-compression  
**Spec Reference:** §3.3

**Constraints:**

- Optional feature

- Only applies when dimensionless

**Guidance:**

- True: Compress large signal ranges

- False: Linear signal accumulation

- Typical: False (for interpretability)

**Example:**

```python
config["use_log_compression"] = False  # No log-compression
```

---

## Threshold Dynamics (§4)

### `c1` - Signal Cost Coefficient

**Type:** Float  
**Valid Range:** (0, ∞)  
**Typical Values:** 0.1-0.5  
**Default:** 0.2  
**Physical Meaning:** Cost of signal processing  
**Spec Reference:** §4.2

**Constraints:**

- Must be positive

- Affects cost computation

**Guidance:**

- Smaller values: Lower cost, higher threshold

- Larger values: Higher cost, lower threshold

- Typical: 0.2 for moderate cost

**Example:**

```python
config["c1"] = 0.2  # Signal cost coefficient
```

---

### `c2` - Ignition Cost Coefficient

**Type:** Float  
**Valid Range:** (0, ∞)  
**Typical Values:** 0.05-0.2  
**Default:** 0.1  
**Physical Meaning:** Cost of ignition event  
**Spec Reference:** §4.2

**Constraints:**

- Must be positive

- Affects refractory period

**Guidance:**

- Smaller values: Lower ignition cost

- Larger values: Higher ignition cost

- Typical: 0.1 for moderate cost

**Example:**

```python
config["c2"] = 0.1  # Ignition cost coefficient
```

---

### `eta` - Learning Rate

**Type:** Float  
**Valid Range:** (0, 1)  
**Typical Values:** 0.01-0.2  
**Default:** 0.1  
**Physical Meaning:** Threshold adaptation rate  
**Spec Reference:** §4.1

**Constraints:**

- Must be positive

- Should be < 1

- Affects learning speed

**Guidance:**

- Smaller values: Slower learning

- Larger values: Faster learning

- Typical: 0.1 for moderate learning

**Example:**

```python
config["eta"] = 0.1  # 10% learning rate
```

---

### `delta_reset` - Refractory Boost

**Type:** Float  
**Valid Range:** (0, ∞)  
**Typical Values:** 0.1-1.0  
**Default:** 0.5  
**Physical Meaning:** Threshold boost after ignition  
**Spec Reference:** §4.1, §6

**Constraints:**

- Must be positive

- Affects refractory period

**Guidance:**

- Smaller values: Shorter refractory period

- Larger values: Longer refractory period

- Typical: 0.5 for moderate refractory

**Example:**

```python
config["delta_reset"] = 0.5  # 50% refractory boost
```

---

### `kappa` - Decay Rate

**Type:** Float  
**Valid Range:** (0, ∞)  
**Typical Values:** 0.01-0.5  
**Default:** 0.15  
**Physical Meaning:** Threshold decay rate  
**Spec Reference:** §4.5, §7.3

**Constraints:**

- Must be positive

- Affects threshold relaxation

**Guidance:**

- Smaller values: Slower decay

- Larger values: Faster decay

- Typical: 0.15 for moderate decay

**Example:**

```python
config["kappa"] = 0.15  # 15% decay rate
```

---

### `theta_base` - Baseline Threshold

**Type:** Float  
**Valid Range:** (0, ∞)  
**Typical Values:** 0.1-1.0  
**Default:** 0.5  
**Physical Meaning:** Resting threshold level  
**Spec Reference:** §4.1

**Constraints:**

- Must be positive

- Affects threshold dynamics

**Guidance:**

- Smaller values: Lower baseline, more ignitions

- Larger values: Higher baseline, fewer ignitions

- Typical: 0.5 for moderate baseline

**Example:**

```python
config["theta_base"] = 0.5  # Baseline threshold 0.5
```

---

## Ignition Mechanism (§5)

### `ignite_tau` - Sigmoid Temperature

**Type:** Float  
**Valid Range:** (0, ∞)  
**Typical Values:** 0.1-1.0  
**Default:** 0.5  
**Physical Meaning:** Sigmoid sharpness for soft ignition  
**Spec Reference:** §5.2

**Constraints:**

- Must be positive

- Affects ignition sharpness

**Guidance:**

- Smaller values: Sharper transition

- Larger values: Smoother transition

- Typical: 0.5 for moderate sharpness

**Example:**

```python
config["ignite_tau"] = 0.5  # Sigmoid temperature 0.5
```

---

### `use_soft_ignition` - Soft vs Hard Ignition

**Type:** Boolean  
**Valid Values:** True, False  
**Default:** False  
**Physical Meaning:** Whether to use soft (stochastic) ignition  
**Spec Reference:** §5.1, §5.2

**Constraints:**

- Optional feature

- Affects ignition behavior

**Guidance:**

- True: Stochastic ignition (probabilistic)

- False: Deterministic ignition (binary)

- Typical: False (for interpretability)

**Example:**

```python
config["use_soft_ignition"] = False  # Hard ignition
```

---

## Post-Ignition Reset (§6)

### `reset_factor` - Signal Reset Factor

**Type:** Float  
**Valid Range:** (0, 1)  
**Typical Values:** 0.1-0.9  
**Default:** 0.5  
**Physical Meaning:** Signal retention after ignition  
**Spec Reference:** §6.1

**Constraints:**

- Must be > 0

- Must be < 1

- Enforced at initialization

**Guidance:**

- Smaller values: More signal reset

- Larger values: Less signal reset

- Typical: 0.5 for moderate reset

**Example:**

```python
config["reset_factor"] = 0.5  # 50% signal retention
```

---

## Continuous-Time SDE (§7)

### `dt` - Time Step

**Type:** Float  
**Valid Range:** (0, min(tau)/10]  
**Typical Values:** 0.1-1.0 ms  
**Default:** 1.0 ms  
**Physical Meaning:** Integration time step  
**Spec Reference:** §7.4

**Constraints:**

- Must be positive

- Must be ≤ min(tau_s, tau_theta, tau_pi) / 10

- Enforced at initialization

**Guidance:**

- Smaller values: More accurate, slower

- Larger values: Less accurate, faster

- Typical: 1.0 ms for 1 kHz sampling

**Example:**

```python
config["dt"] = 1.0  # 1 ms time step
```

---

### `sigma_s` - Signal Noise Amplitude

**Type:** Float  
**Valid Range:** (0, ∞)  
**Typical Values:** 0.01-0.5  
**Default:** 0.1  
**Physical Meaning:** Noise amplitude for signal SDE  
**Spec Reference:** §7.1, §7.2

**Constraints:**

- Must be positive

- Affects stochasticity

**Guidance:**

- Smaller values: Less noise

- Larger values: More noise

- Typical: 0.1 for moderate noise

**Example:**

```python
config["sigma_s"] = 0.1  # Signal noise amplitude
```

---

## Hierarchical Architecture (§8)

### `use_hierarchy` - Enable Hierarchical System

**Type:** Boolean  
**Valid Values:** True, False  
**Default:** False  
**Physical Meaning:** Whether to use hierarchical dynamics  
**Spec Reference:** §8

**Constraints:**

- Optional feature

- Requires n_levels and timescale_k

**Guidance:**

- True: Multi-scale processing

- False: Single-scale processing

- Typical: False (for simplicity)

**Example:**

```python
config["use_hierarchy"] = False  # No hierarchical system
```

---

### `n_levels` - Number of Hierarchical Levels

**Type:** Integer  
**Valid Range:** [1, ∞)  
**Typical Values:** 3-7  
**Default:** 5  
**Physical Meaning:** Number of timescale levels  
**Spec Reference:** §8.1

**Constraints:**

- Must be ≥ 1

- Only used if use_hierarchy is True

**Guidance:**

- Smaller values: Fewer timescales

- Larger values: More timescales

- Typical: 5 for cortical hierarchy

**Example:**

```python
config["n_levels"] = 5  # 5 hierarchical levels
```

---

### `timescale_k` - Timescale Ratio

**Type:** Float  
**Valid Range:** (1, ∞)  
**Typical Values:** 1.5-2.0  
**Default:** 1.6  
**Physical Meaning:** Geometric ratio between timescales  
**Spec Reference:** §8.1

**Constraints:**

- Must be > 1

- Enforced at initialization

- Only used if use_hierarchy is True

**Guidance:**

- Smaller values: Closer timescales

- Larger values: More separated timescales

- Typical: 1.6 for cortical hierarchy

**Example:**

```python
config["timescale_k"] = 1.6  # Timescale ratio 1.6
```

---

## Oscillatory Coupling (§9)

### `use_kuramoto` - Enable Kuramoto Oscillators

**Type:** Boolean  
**Valid Values:** True, False  
**Default:** False  
**Physical Meaning:** Whether to use Kuramoto oscillators  
**Spec Reference:** §9

**Constraints:**

- Optional feature

- Requires kuramoto_tau_xi and kuramoto_sigma_xi

**Guidance:**

- True: Coupled phase dynamics

- False: No oscillatory coupling

- Typical: False (for simplicity)

**Example:**

```python
config["use_kuramoto"] = False  # No Kuramoto oscillators
```

---

### `kuramoto_tau_xi` - OU Noise Timescale

**Type:** Float  
**Valid Range:** (0, ∞)  
**Typical Values:** 0.1-10.0 ms  
**Default:** 1.0 ms  
**Physical Meaning:** Ornstein-Uhlenbeck noise timescale  
**Spec Reference:** §9

**Constraints:**

- Must be positive

- Only used if use_kuramoto is True

**Guidance:**

- Smaller values: Faster noise

- Larger values: Slower noise

- Typical: 1.0 ms for neural timescales

**Example:**

```python
config["kuramoto_tau_xi"] = 1.0  # 1 ms OU timescale
```

---

### `kuramoto_sigma_xi` - OU Noise Amplitude

**Type:** Float  
**Valid Range:** (0, ∞)  
**Typical Values:** 0.01-0.5 rad/ms  
**Default:** 0.1 rad/ms  
**Physical Meaning:** Ornstein-Uhlenbeck noise amplitude  
**Spec Reference:** §9

**Constraints:**

- Must be positive

- Only used if use_kuramoto is True

**Guidance:**

- Smaller values: Less noise

- Larger values: More noise

- Typical: 0.1 for moderate noise

**Example:**

```python
config["kuramoto_sigma_xi"] = 0.1  # OU noise amplitude
```

---

## Reservoir Layer (§10)

### `use_reservoir` - Enable Reservoir Layer

**Type:** Boolean  
**Valid Values:** True, False  
**Default:** False  
**Physical Meaning:** Whether to use reservoir computing  
**Spec Reference:** §10

**Constraints:**

- Optional feature

- Requires reservoir_size and reservoir_tau

**Guidance:**

- True: Liquid state machine

- False: No reservoir layer

- Typical: False (for simplicity)

**Example:**

```python
config["use_reservoir"] = False  # No reservoir layer
```

---

### `reservoir_size` - Reservoir Neuron Count

**Type:** Integer  
**Valid Range:** [10, 10000]  
**Typical Values:** 50-500  
**Default:** 100  
**Physical Meaning:** Number of reservoir neurons  
**Spec Reference:** §10

**Constraints:**

- Must be ≥ 10

- Only used if use_reservoir is True

- Affects memory and computation

**Guidance:**

- Smaller values: Faster, less capacity

- Larger values: Slower, more capacity

- Typical: 100 for moderate capacity

**Example:**

```python
config["reservoir_size"] = 100  # 100 reservoir neurons
```

---

### `reservoir_tau` - Reservoir Timescale

**Type:** Float  
**Valid Range:** (0, ∞)  
**Typical Values:** 0.1-10.0 ms  
**Default:** 1.0 ms  
**Physical Meaning:** Reservoir state timescale  
**Spec Reference:** §10

**Constraints:**

- Must be positive

- Only used if use_reservoir is True

**Guidance:**

- Smaller values: Faster dynamics

- Larger values: Slower dynamics

- Typical: 1.0 ms for neural timescales

**Example:**

```python
config["reservoir_tau"] = 1.0  # 1 ms reservoir timescale
```

---

## Thermodynamic Constraints (§11)

### `use_thermodynamics` - Enable Landauer Cost

**Type:** Boolean  
**Valid Values:** True, False  
**Default:** False  
**Physical Meaning:** Whether to compute Landauer cost  
**Spec Reference:** §11

**Constraints:**

- Optional feature

- Requires k_boltzmann, T_env, kappa_meta

**Guidance:**

- True: Thermodynamic grounding

- False: No thermodynamic cost

- Typical: False (for simplicity)

**Example:**

```python
config["use_thermodynamics"] = False  # No thermodynamic cost
```

---

### `k_boltzmann` - Boltzmann Constant

**Type:** Float  
**Valid Range:** (0, ∞)  
**Typical Values:** 1.38e-23 J/K  
**Default:** 1.38e-23 J/K  
**Physical Meaning:** Boltzmann constant  
**Spec Reference:** §11

**Constraints:**

- Must be positive

- Physical constant (do not change)

- Only used if use_thermodynamics is True

**Guidance:**

- Use default value (physical constant)

- Do not modify

**Example:**

```python
config["k_boltzmann"] = 1.38e-23  # Boltzmann constant
```

---

### `T_env` - Environment Temperature

**Type:** Float  
**Valid Range:** (0, ∞)  
**Typical Values:** 273-373 K  
**Default:** 310.0 K  
**Physical Meaning:** Environment temperature (Kelvin)  
**Spec Reference:** §11

**Constraints:**

- Must be positive

- Typical: 310 K (37°C, body temperature)

- Only used if use_thermodynamics is True

**Guidance:**

- Use 310 K for biological systems

- Adjust for different environments

**Example:**

```python
config["T_env"] = 310.0  # 37°C body temperature
```

---

### `kappa_meta` - Metabolic Efficiency

**Type:** Float  
**Valid Range:** (0, ∞)  
**Typical Values:** 0.5-2.0  
**Default:** 1.0  
**Physical Meaning:** Metabolic efficiency factor  
**Spec Reference:** §11

**Constraints:**

- Must be positive

- Affects thermodynamic cost scaling

- Only used if use_thermodynamics is True

**Guidance:**

- 1.0: Ideal efficiency

- < 1.0: Better than ideal

- > 1.0: Worse than ideal

- Typical: 1.0 for baseline

**Example:**

```python
config["kappa_meta"] = 1.0  # Ideal metabolic efficiency
```

---

## Observable Mapping (§14)

### `use_observable_mapping` - Enable Observable Extraction

**Type:** Boolean  
**Valid Values:** True, False  
**Default:** False  
**Physical Meaning:** Whether to extract observables  
**Spec Reference:** §14

**Constraints:**

- Optional feature

- Enables empirical validation

**Guidance:**

- True: Extract neural/behavioral observables

- False: No observable extraction

- Typical: False (for simplicity)

**Example:**

```python
config["use_observable_mapping"] = False  # No observable extraction
```

---

## Stability Analysis (§7)

### `use_stability_analysis` - Enable Stability Analysis

**Type:** Boolean  
**Valid Values:** True, False  
**Default:** False  
**Physical Meaning:** Whether to analyze stability  
**Spec Reference:** §7

**Constraints:**

- Optional feature

- Enables theoretical validation

**Guidance:**

- True: Compute eigenvalues and stability

- False: No stability analysis

- Typical: False (for speed)

**Example:**

```python
config["use_stability_analysis"] = False  # No stability analysis
```

---

## Validation Summary

All parameters are validated at pipeline initialization. Invalid configurations raise `ValueError` with descriptive messages.

**Validation Categories:**

1. ✅ Neuromodulator separation (NE mutual exclusivity)
2. ✅ Signal accumulation (lam ∈ (0,1))
3. ✅ Threshold dynamics (kappa > 0, reset_factor ∈ (0,1))
4. ✅ Ignition dynamics (ignite_tau > 0)
5. ✅ Continuous-time SDE (dt ≤ min(tau)/10)
6. ✅ Hierarchical parameters (k > 1)
7. ✅ Precision parameters (pi_min < pi_max)
8. ✅ Numerical stability (eps_stab > 0)

---

## Quick Reference

### Minimal Configuration

```python
config = {
    "tau_s": 5.0,
    "tau_pi": 1000.0,
    "tau_theta": 1000.0,
    "lam": 0.2,
    "eta": 0.1,
    "dt": 1.0,
}
```

### Full Configuration

```python
from config import CONFIG
config = CONFIG.copy()
```

### Validation

```python
from core.validation import validate_config
validate_config(config)  # Raises ValueError if invalid
```

---

## References

- APGI Specification: `APGI-Specs.md`

- API Reference: `docs/API_REFERENCE.md`

- Design Choices: `docs/DESIGN_CHOICES.md`

- Troubleshooting: `docs/TROUBLESHOOTING.md`
