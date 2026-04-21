# APGI Troubleshooting Guide

**Version:** 1.0  
**Date:** April 21, 2026  
**Status:** Production Ready

---

## Common Errors

### Error: `ValueError: lam must be in (0,1)`

**Cause:** Integration rate parameter is outside valid range.

**Solution:**
```python
# Wrong
config["lam"] = 0.0  # Invalid: must be > 0
config["lam"] = 1.0  # Invalid: must be < 1

# Correct
config["lam"] = 0.2  # Valid: between 0 and 1
```

**Reference:** `docs/PARAMETER_CONSTRAINTS.md` - Signal Accumulation

---

### Error: `ValueError: NE cannot modulate both precision and threshold`

**Cause:** Norepinephrine is configured to modulate both precision and threshold simultaneously.

**Solution:**
```python
# Wrong
config["ne_on_precision"] = True
config["ne_on_threshold"] = True  # Conflict!

# Correct (Option 1: NE modulates precision)
config["ne_on_precision"] = True
config["ne_on_threshold"] = False

# Correct (Option 2: NE modulates threshold)
config["ne_on_precision"] = False
config["ne_on_threshold"] = True
```

**Reference:** `docs/PARAMETER_CONSTRAINTS.md` - Precision System

---

### Error: `ValueError: dt=X exceeds max Y`

**Cause:** Time step is too large for numerical stability.

**Solution:**
```python
# Check current dt and timescales
dt = config["dt"]
tau_s = config["tau_s"]
tau_theta = config["tau_theta"]
tau_pi = config["tau_pi"]

min_tau = min(tau_s, tau_theta, tau_pi)
max_dt = min_tau / 10

print(f"Current dt: {dt}")
print(f"Max allowed dt: {max_dt}")

# Fix: Reduce dt
config["dt"] = max_dt * 0.9  # Use 90% of max
```

**Reference:** `docs/PARAMETER_CONSTRAINTS.md` - Continuous-Time SDE

---

### Error: `ValueError: k must be > 1`

**Cause:** Hierarchical timescale ratio is invalid.

**Solution:**
```python
# Wrong
config["timescale_k"] = 1.0  # Invalid: must be > 1
config["timescale_k"] = 0.5  # Invalid: must be > 1

# Correct
config["timescale_k"] = 1.6  # Valid: typical cortical ratio
```

**Reference:** `docs/PARAMETER_CONSTRAINTS.md` - Hierarchical Architecture

---

### Error: `ValueError: reset_factor must be in (0,1)`

**Cause:** Signal reset factor is outside valid range.

**Solution:**
```python
# Wrong
config["reset_factor"] = 0.0  # Invalid: must be > 0
config["reset_factor"] = 1.0  # Invalid: must be < 1

# Correct
config["reset_factor"] = 0.5  # Valid: 50% signal retention
```

**Reference:** `docs/PARAMETER_CONSTRAINTS.md` - Post-Ignition Reset

---

## Performance Issues

### Issue: Pipeline is slow

**Diagnosis:**
```python
import time
from pipeline import APGIPipeline
from config import CONFIG

pipeline = APGIPipeline(CONFIG)

# Time a single step
start = time.time()
for _ in range(1000):
    pipeline.step(0.5, 0.3, 0.2, 0.1)
elapsed = time.time() - start

print(f"Time per step: {elapsed/1000*1000:.2f} ms")
```

**Solutions:**

1. **Disable optional features:**
```python
config["use_kuramoto"] = False
config["use_reservoir"] = False
config["use_observable_mapping"] = False
config["use_stability_analysis"] = False
```

2. **Increase time step (if acceptable):**
```python
# Check max allowed dt
min_tau = min(config["tau_s"], config["tau_theta"], config["tau_pi"])
max_dt = min_tau / 10
config["dt"] = max_dt * 0.9  # Use larger dt
```

3. **Reduce hierarchical levels:**
```python
config["n_levels"] = 3  # Fewer levels = faster
```

4. **Reduce reservoir size:**
```python
config["reservoir_size"] = 50  # Smaller reservoir = faster
```

---

### Issue: Memory usage is high

**Diagnosis:**
```python
import tracemalloc
from pipeline import APGIPipeline
from config import CONFIG

tracemalloc.start()

pipeline = APGIPipeline(CONFIG)

# Run simulation
for _ in range(10000):
    pipeline.step(0.5, 0.3, 0.2, 0.1)

current, peak = tracemalloc.get_traced_memory()
print(f"Current memory: {current / 1024 / 1024:.1f} MB")
print(f"Peak memory: {peak / 1024 / 1024:.1f} MB")
```

**Solutions:**

1. **Disable history recording:**
```python
# Modify pipeline to not store history
# (requires code change)
```

2. **Reduce reservoir size:**
```python
config["reservoir_size"] = 50  # Smaller = less memory
```

3. **Reduce hierarchical levels:**
```python
config["n_levels"] = 3  # Fewer levels = less memory
```

---

## Numerical Issues

### Issue: Signal becomes NaN or Inf

**Cause:** Numerical instability due to large values or division by zero.

**Solutions:**

1. **Increase stability constant:**
```python
config["eps_stab"] = 1e-6  # Larger value = more stable
```

2. **Reduce time step:**
```python
config["dt"] = 0.1  # Smaller dt = more stable
```

3. **Clamp precision bounds:**
```python
config["pi_min"] = 0.001  # Smaller minimum
config["pi_max"] = 1000.0  # Larger maximum
```

4. **Check input ranges:**
```python
# Ensure inputs are reasonable
assert -10 < x_e < 10, "Exteroceptive signal out of range"
assert -10 < x_i < 10, "Interoceptive signal out of range"
```

---

### Issue: Threshold diverges

**Cause:** Learning rate too high or cost/value imbalance.

**Solutions:**

1. **Reduce learning rate:**
```python
config["eta"] = 0.01  # Smaller = slower learning
```

2. **Adjust cost coefficients:**
```python
config["c1"] = 0.1  # Reduce signal cost
config["c2"] = 0.05  # Reduce ignition cost
```

3. **Check stability:**
```python
from analysis.stability import StabilityAnalyzer

analyzer = StabilityAnalyzer(config)
result = analyzer.check_stability()
print(f"Stable: {result['stable']}")
print(f"Max eigenvalue: {result['max_eigenvalue']:.4f}")
```

---

## Validation Issues

### Issue: Configuration validation fails

**Diagnosis:**
```python
from core.validation import validate_config

try:
    validate_config(config)
except ValueError as e:
    print(f"Validation error: {e}")
```

**Solutions:**

1. **Check parameter ranges:**
```python
from docs.PARAMETER_CONSTRAINTS import *  # See guide
```

2. **Use default configuration:**
```python
from config import CONFIG
config = CONFIG.copy()  # Start with valid defaults
```

3. **Validate step-by-step:**
```python
from core.validation import validate_parameter

# Check individual parameters
validate_parameter("lam", config["lam"], 0, 1)
validate_parameter("eta", config["eta"], 0, 1)
validate_parameter("dt", config["dt"], 0, float('inf'))
```

---

## Observable Mapping Issues

### Issue: Observable values are NaN

**Cause:** Insufficient history or invalid signal values.

**Solutions:**

1. **Ensure sufficient history:**
```python
# Run pipeline for several steps before extracting observables
for _ in range(100):
    pipeline.step(0.5, 0.3, 0.2, 0.1)

# Now extract observables
result = pipeline.step(0.5, 0.3, 0.2, 0.1)
print(result["neural_gamma_power"])
```

2. **Check signal ranges:**
```python
# Ensure signals are in reasonable range
assert 0 < result["S"] < 10, "Signal out of range"
assert 0 < result["theta"] < 10, "Threshold out of range"
```

---

### Issue: Observable values are constant

**Cause:** Signal or threshold not changing.

**Solutions:**

1. **Vary input signals:**
```python
# Use varying inputs instead of constant
for t in range(1000):
    x_e = 0.5 + 0.1 * np.sin(2 * np.pi * t / 100)
    x_i = 0.2 + 0.05 * np.cos(2 * np.pi * t / 100)
    result = pipeline.step(x_e, 0.3, x_i, 0.1)
```

2. **Check neuromodulator inputs:**
```python
# Vary neuromodulators
for t in range(1000):
    g_ach = 1.0 + 0.2 * np.sin(2 * np.pi * t / 100)
    g_ne = 0.5 * np.cos(2 * np.pi * t / 100)
    result = pipeline.step(0.5, 0.3, 0.2, 0.1, g_ach, g_ne)
```

---

## Stability Analysis Issues

### Issue: System is unstable

**Diagnosis:**
```python
from analysis.stability import StabilityAnalyzer

analyzer = StabilityAnalyzer(config)
result = analyzer.check_stability()

if not result["stable"]:
    print(f"Unstable! Max eigenvalue: {result['max_eigenvalue']:.4f}")
    print(f"Eigenvalues: {result['eigenvalues']}")
```

**Solutions:**

1. **Reduce learning rate:**
```python
config["eta"] = 0.01  # Smaller = more stable
```

2. **Increase decay rate:**
```python
config["kappa"] = 0.3  # Larger = faster decay
```

3. **Adjust cost/value balance:**
```python
config["c1"] = 0.1  # Reduce signal cost
config["c2"] = 0.05  # Reduce ignition cost
```

---

## Testing & Debugging

### Test Configuration Validity

```python
from core.validation import validate_config
from config import CONFIG

try:
    validate_config(CONFIG)
    print("✓ Configuration is valid")
except ValueError as e:
    print(f"✗ Configuration error: {e}")
```

### Test Pipeline Initialization

```python
from pipeline import APGIPipeline
from config import CONFIG

try:
    pipeline = APGIPipeline(CONFIG)
    print("✓ Pipeline initialized successfully")
except Exception as e:
    print(f"✗ Pipeline initialization error: {e}")
```

### Test Single Step

```python
from pipeline import APGIPipeline
from config import CONFIG

pipeline = APGIPipeline(CONFIG)

try:
    result = pipeline.step(0.5, 0.3, 0.2, 0.1)
    print("✓ Single step executed successfully")
    print(f"  Signal: {result['S']:.4f}")
    print(f"  Threshold: {result['theta']:.4f}")
    print(f"  Ignition: {result['B']}")
except Exception as e:
    print(f"✗ Step execution error: {e}")
```

### Test Full Simulation

```python
from pipeline import APGIPipeline
from config import CONFIG
import numpy as np

pipeline = APGIPipeline(CONFIG)

try:
    for t in range(1000):
        x_e = 0.5 + 0.1 * np.sin(2 * np.pi * t / 100)
        x_i = 0.2 + 0.05 * np.cos(2 * np.pi * t / 100)
        result = pipeline.step(x_e, 0.3, x_i, 0.1)
    
    print("✓ Full simulation completed successfully")
except Exception as e:
    print(f"✗ Simulation error: {e}")
```

---

## FAQ

### Q: What are typical parameter values?

**A:** See `docs/PARAMETER_CONSTRAINTS.md` for typical values for each parameter.

---

### Q: How do I enable all features?

**A:** 
```python
config = CONFIG.copy()
config["use_kuramoto"] = True
config["use_reservoir"] = True
config["use_observable_mapping"] = True
config["use_stability_analysis"] = True
config["use_thermodynamics"] = True
config["use_hierarchy"] = True
```

---

### Q: How do I disable all optional features?

**A:**
```python
config = CONFIG.copy()
config["use_kuramoto"] = False
config["use_reservoir"] = False
config["use_observable_mapping"] = False
config["use_stability_analysis"] = False
config["use_thermodynamics"] = False
config["use_hierarchy"] = False
```

---

### Q: What's the fastest configuration?

**A:**
```python
config = CONFIG.copy()
# Disable all optional features
config["use_kuramoto"] = False
config["use_reservoir"] = False
config["use_observable_mapping"] = False
config["use_stability_analysis"] = False
config["use_thermodynamics"] = False
config["use_hierarchy"] = False

# Use larger time step (if acceptable)
config["dt"] = 0.5  # Larger dt = faster
```

---

### Q: What's the most accurate configuration?

**A:**
```python
config = CONFIG.copy()
# Use smaller time step
config["dt"] = 0.1  # Smaller dt = more accurate

# Use smaller stability constant
config["eps_stab"] = 1e-10  # More accurate

# Enable all features
config["use_kuramoto"] = True
config["use_reservoir"] = True
config["use_observable_mapping"] = True
config["use_stability_analysis"] = True
config["use_thermodynamics"] = True
config["use_hierarchy"] = True
```

---

### Q: How do I run the test suite?

**A:**
```bash
python -m pytest tests/ -v
```

---

### Q: How do I check code coverage?

**A:**
```bash
python -m pytest tests/ --cov=. --cov-report=html
```

---

## Getting Help

1. **Check documentation:**
   - `docs/API_REFERENCE.md` - Function documentation
   - `docs/DESIGN_CHOICES.md` - Design rationale
   - `docs/PARAMETER_CONSTRAINTS.md` - Parameter guide

2. **Run tests:**
   ```bash
   python -m pytest tests/ -v
   ```

3. **Check examples:**
   - `examples/01_basic_usage.py`
   - `examples/02_advanced_features.py`
   - `examples/03_observable_mapping.py`
   - `examples/04_thermodynamics.py`

4. **Review specification:**
   - `APGI-Specs.md` - Full specification
   - `PHASE_2_IMPLEMENTATION_SUMMARY.md` - Implementation details

---

## References

- API Reference: `docs/API_REFERENCE.md`
- Parameter Constraints: `docs/PARAMETER_CONSTRAINTS.md`
- Design Choices: `docs/DESIGN_CHOICES.md`
- Examples: `examples/`
- Specification: `APGI-Specs.md`

