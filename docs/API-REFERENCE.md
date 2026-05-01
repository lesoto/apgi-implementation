# APGI API Reference

## Core Pipeline

### `APGIPipeline`

Main execution pipeline for APGI system.

```python
class APGIPipeline:
    """APGI unified pipeline per specification.
    
    Implements all 15 specification sections with optional features.
    
    Spec Reference: §13 Execution Pipeline
    """
    
    def __init__(self, config: dict):
        """Initialize APGI pipeline.
        
        Args:
            config: Configuration dictionary with all parameters
            
        Raises:
            ValueError: If configuration violates constraints
            
        Example:
            >>> from config import CONFIG
            >>> pipeline = APGIPipeline(CONFIG)
        """
    
    def step(
        self,
        x_e: float,
        x_hat_e: float,
        x_i: float,
        x_hat_i: float,
        g_ach: float = 1.0,
        g_ne: float = 0.0,
        beta_da: float = 0.0
    ) -> dict:
        """Execute single APGI step.
        
        Args:
            x_e: Exteroceptive prediction error
            x_hat_e: Exteroceptive prediction
            x_i: Interoceptive prediction error
            x_hat_i: Interoceptive prediction
            g_ach: Acetylcholine gain (0-1)
            g_ne: Norepinephrine gain (0-1)
            beta_da: Dopamine bias (-1 to 1)
            
        Returns:
            Dictionary with all computed values:
            - S: Signal magnitude
            - theta: Threshold
            - B: Ignition binary
            - P_ign: Soft ignition probability
            - delta: Ignition margin
            - cost: Metabolic cost
            - value: Information value
            - thermodynamic_cost: Landauer cost
            - neural_gamma_power: Gamma-band power (if observable mapping)
            - behavioral_rt_variability: RT variability (if observable mapping)
            - kuramoto_phases: Phase values (if Kuramoto enabled)
            - stability_eigenvalues: Eigenvalues (if stability analysis)
            
        Spec Reference: §13 Execution Pipeline
        
        Example:
            >>> result = pipeline.step(x_e=0.5, x_hat_e=0.3, x_i=0.2, x_hat_i=0.1)
            >>> print(f"Signal: {result['S']:.4f}")
            >>> print(f"Threshold: {result['theta']:.4f}")
            >>> print(f"Ignition: {result['B']}")
        """
    
    def reset(self):
        """Reset pipeline state to initial conditions.
        
        Resets:
        - Signal accumulator
        - Threshold
        - Precision values
        - Neuromodulator history
        - All optional components
        
        Example:
            >>> pipeline.reset()
        """
    
    def get_state(self) -> dict:
        """Get current pipeline state.
        
        Returns:
            Dictionary with all state variables
            
        Example:
            >>> state = pipeline.get_state()
            >>> print(state.keys())
        """
    
    def set_state(self, state: dict):
        """Set pipeline state.
        
        Args:
            state: State dictionary from get_state()
            
        Example:
            >>> state = pipeline.get_state()
            >>> pipeline.set_state(state)
        """
```

---

## Signal Processing

### `compute_prediction_errors`

Compute raw prediction errors.

```python
def compute_prediction_errors(
    x_e: float,
    x_hat_e: float,
    x_i: float,
    x_hat_i: float
) -> tuple[float, float]:
    """Compute prediction errors.
    
    ε_e(t) = x_e(t) - x̂_e(t)
    ε_i(t) = x_i(t) - x̂_i(t)
    
    Args:
        x_e: Exteroceptive signal
        x_hat_e: Exteroceptive prediction
        x_i: Interoceptive signal
        x_hat_i: Interoceptive prediction
        
    Returns:
        (eps_e, eps_i): Prediction errors
        
    Spec Reference: §1.1 Raw Prediction Errors
    
    Example:
        >>> eps_e, eps_i = compute_prediction_errors(0.5, 0.3, 0.2, 0.1)
        >>> print(f"Exteroceptive error: {eps_e}")
        >>> print(f"Interoceptive error: {eps_i}")
    """
```

### `compute_z_scores`

Normalize prediction errors to z-scores.

```python
def compute_z_scores(
    eps_e: float,
    eps_i: float,
    mu_e: float,
    mu_i: float,
    sigma_e: float,
    sigma_i: float,
    eps_stab: float = 1e-8
) -> tuple[float, float]:
    """Compute z-score normalized errors.
    
    z_e = (ε_e - μ_e) / (σ_e + ε_stab)
    z_i = (ε_i - μ_i) / (σ_i + ε_stab)
    
    Args:
        eps_e: Exteroceptive prediction error
        eps_i: Interoceptive prediction error
        mu_e: Exteroceptive mean
        mu_i: Interoceptive mean
        sigma_e: Exteroceptive std
        sigma_i: Interoceptive std
        eps_stab: Stability constant (default 1e-8)
        
    Returns:
        (z_e, z_i): Z-score normalized errors
        
    Spec Reference: §1.3 Z-Score Normalization
    
    Example:
        >>> z_e, z_i = compute_z_scores(0.2, 0.1, 0.0, 0.0, 0.1, 0.1)
        >>> print(f"z_e: {z_e:.4f}, z_i: {z_i:.4f}")
    """
```

---

## Precision System

### `compute_precision`

Compute precision from variance.

```python
def compute_precision(
    sigma: float,
    pi_min: float = 0.01,
    pi_max: float = 100.0,
    eps_stab: float = 1e-8
) -> float:
    """Compute precision with clamping.
    
    Π = 1 / (σ² + ε_stab)
    Π ∈ [Π_min, Π_max]
    
    Args:
        sigma: Standard deviation
        pi_min: Minimum precision (default 0.01)
        pi_max: Maximum precision (default 100.0)
        eps_stab: Stability constant (default 1e-8)
        
    Returns:
        Precision value (clamped)
        
    Spec Reference: §2.2 Precision Definition
    
    Example:
        >>> pi = compute_precision(0.1)
        >>> print(f"Precision: {pi:.4f}")
    """
```

### `apply_neuromodulatory_gains`

Apply neuromodulatory modulation to precision and signals.

```python
def apply_neuromodulatory_gains(
    pi_e: float,
    pi_i: float,
    z_i: float,
    g_ach: float = 1.0,
    g_ne: float = 0.0,
    beta_da: float = 0.0
) -> tuple[float, float, float]:
    """Apply neuromodulatory gains.
    
    Π_e^eff = g_ACh · Π_e
    Π_i^eff = g_NE · Π_i
    z_i^eff = z_i + β_DA
    
    Args:
        pi_e: Exteroceptive precision
        pi_i: Interoceptive precision
        z_i: Interoceptive z-score
        g_ach: Acetylcholine gain (0-1)
        g_ne: Norepinephrine gain (0-1)
        beta_da: Dopamine bias (-1 to 1)
        
    Returns:
        (pi_e_eff, pi_i_eff, z_i_eff): Modulated values
        
    Spec Reference: §2.3 Neuromodulatory Gains
    
    Example:
        >>> pi_e_eff, pi_i_eff, z_i_eff = apply_neuromodulatory_gains(
        ...     pi_e=10.0, pi_i=5.0, z_i=0.5,
        ...     g_ach=1.2, g_ne=0.8, beta_da=0.1
        ... )
    """
```

---

## Threshold Dynamics

### `compute_metabolic_cost`

Compute metabolic cost of signal processing.

```python
def compute_metabolic_cost(
    S: float,
    B_prev: float,
    c1: float = 0.2,
    c2: float = 0.1
) -> float:
    """Compute metabolic cost.
    
    C(t) = c₁·S(t) + c₂·B(t-1)
    
    Args:
        S: Signal magnitude
        B_prev: Previous ignition state
        c1: Signal cost coefficient (default 0.2)
        c2: Ignition cost coefficient (default 0.1)
        
    Returns:
        Metabolic cost
        
    Spec Reference: §4.2 Metabolic Cost
    
    Example:
        >>> cost = compute_metabolic_cost(S=0.5, B_prev=1.0)
        >>> print(f"Cost: {cost:.4f}")
    """
```

### `compute_information_value`

Compute information value of signals.

```python
def compute_information_value(
    z_e: float,
    z_i_eff: float,
    v1: float = 0.3,
    v2: float = 0.2
) -> float:
    """Compute information value.
    
    V(t) = v₁|z_e| + v₂|z_i^eff|
    
    Args:
        z_e: Exteroceptive z-score
        z_i_eff: Modulated interoceptive z-score
        v1: Exteroceptive value weight (default 0.3)
        v2: Interoceptive value weight (default 0.2)
        
    Returns:
        Information value
        
    Spec Reference: §4.3 Information Value
    
    Example:
        >>> value = compute_information_value(z_e=0.5, z_i_eff=0.3)
        >>> print(f"Value: {value:.4f}")
    """
```

### `update_threshold`

Update threshold via allostatic mechanism.

```python
def update_threshold(
    theta_prev: float,
    S: float,
    B_prev: float,
    cost: float,
    value: float,
    eta: float = 0.1,
    delta_reset: float = 0.5,
    kappa: float = 0.15,
    theta_base: float = 0.5,
    g_ne: float = 0.0,
    gamma_ne: float = 0.2
) -> float:
    """Update threshold.
    
    θ(t+1) = θ(t) + η[C(t) - V(t)] + δ_reset·B(t)
    θ(t+1) = θ_base + (θ(t) - θ_base)·exp(-κ)
    θ ← θ·(1 + γ_NE·g_NE(t))
    
    Args:
        theta_prev: Previous threshold
        S: Signal magnitude
        B_prev: Previous ignition state
        cost: Metabolic cost
        value: Information value
        eta: Learning rate (default 0.1)
        delta_reset: Refractory boost (default 0.5)
        kappa: Decay rate (default 0.15)
        theta_base: Baseline threshold (default 0.5)
        g_ne: Norepinephrine gain (default 0.0)
        gamma_ne: NE modulation strength (default 0.2)
        
    Returns:
        Updated threshold
        
    Spec Reference: §4 Dynamic Ignition Threshold
    
    Example:
        >>> theta_new = update_threshold(
        ...     theta_prev=0.5, S=0.6, B_prev=0, cost=0.1, value=0.2
        ... )
    """
```

---

## Ignition Mechanism

### `compute_ignition_hard`

Hard threshold ignition.

```python
def compute_ignition_hard(S: float, theta: float) -> int:
    """Compute hard ignition.
    
    B(t) = 1 if S(t) > θ(t), else 0
    
    Args:
        S: Signal magnitude
        theta: Threshold
        
    Returns:
        Binary ignition (0 or 1)
        
    Spec Reference: §5.1 Hard Threshold
    
    Example:
        >>> B = compute_ignition_hard(S=0.6, theta=0.5)
        >>> print(f"Ignition: {B}")
    """
```

### `compute_ignition_soft`

Soft threshold ignition with sigmoid.

```python
def compute_ignition_soft(
    S: float,
    theta: float,
    tau_sigma: float = 0.5
) -> float:
    """Compute soft ignition probability.
    
    P_ign = σ([S - θ] / τ_σ)
    
    Args:
        S: Signal magnitude
        theta: Threshold
        tau_sigma: Sigmoid temperature (default 0.5)
        
    Returns:
        Ignition probability (0-1)
        
    Spec Reference: §5.2 Soft Threshold
    
    Example:
        >>> p_ign = compute_ignition_soft(S=0.6, theta=0.5)
        >>> print(f"P(ignition): {p_ign:.4f}")
    """
```

### `compute_ignition_margin`

Compute ignition margin.

```python
def compute_ignition_margin(S: float, theta: float) -> float:
    """Compute ignition margin.
    
    Δ(t) = S(t) - θ(t)
    
    Args:
        S: Signal magnitude
        theta: Threshold
        
    Returns:
        Ignition margin
        
    Spec Reference: §5.3 Ignition Margin
    
    Example:
        >>> delta = compute_ignition_margin(S=0.6, theta=0.5)
        >>> print(f"Margin: {delta:.4f}")
    """
```

---

## Oscillatory Coupling

### `KuramotoOscillators`

Coupled phase oscillators.

```python
class KuramotoOscillators:
    """Kuramoto oscillators per spec §9.
    
    Implements: dφ_ℓ/dt = ω_ℓ + Σ_j K_{ℓj} sin(φ_j - φ_ℓ) + ξ_ℓ(t)
    """
    
    def __init__(
        self,
        n_levels: int,
        omega: np.ndarray,
        K: np.ndarray,
        tau_xi: float = 1.0,
        sigma_xi: float = 0.1
    ):
        """Initialize Kuramoto oscillators.
        
        Args:
            n_levels: Number of oscillators
            omega: Natural frequencies (n_levels,)
            K: Coupling matrix (n_levels, n_levels)
            tau_xi: OU noise timescale (default 1.0)
            sigma_xi: OU noise amplitude (default 0.1)
        """
    
    def step(self, dt: float) -> dict:
        """Update phases.
        
        Args:
            dt: Time step
            
        Returns:
            Dictionary with:
            - phases: Current phase values
            - synchronization: Kuramoto order parameter
            - coherence: Phase coherence matrix
        """
    
    def reset_phase_on_ignition(self, level: int, reset_amount: float = np.pi):
        """Reset phase on ignition event.
        
        Args:
            level: Level to reset
            reset_amount: Reset amount in radians
        """
    
    def get_synchronization_order(self) -> float:
        """Get Kuramoto synchronization order parameter.
        
        Returns:
            Order parameter R ∈ [0, 1]
        """
```

---

## Reservoir Layer

### `LiquidStateMachine`

Reservoir computing layer.

```python
class LiquidStateMachine:
    """Liquid state machine per spec §10.
    
    Implements fixed recurrent network with trained linear readout.
    """
    
    def __init__(
        self,
        N: int = 100,
        M: int = 2,
        tau_res: float = 1.0,
        spectral_radius: float = 0.9,
        input_scale: float = 0.1
    ):
        """Initialize reservoir.
        
        Args:
            N: Reservoir size (default 100)
            M: Input dimension (default 2)
            tau_res: Reservoir timescale (default 1.0)
            spectral_radius: Spectral radius (default 0.9)
            input_scale: Input weight scale (default 0.1)
        """
    
    def step(self, u: np.ndarray, S_margin: float = 0.0) -> float:
        """Update reservoir and compute readout.
        
        Args:
            u: Input vector [z_e, z_i]
            S_margin: Ignition margin Δ(t)
            
        Returns:
            Reservoir readout
        """
    
    def train_readout(
        self,
        X: np.ndarray,
        y: np.ndarray,
        alpha: float = 1e-6
    ):
        """Train linear readout via ridge regression.
        
        Args:
            X: Reservoir states (T, N)
            y: Target signal (T,)
            alpha: Ridge regularization (default 1e-6)
        """
```

---

## Thermodynamics

### `compute_landauer_cost`

Compute thermodynamic cost per Landauer's principle.

```python
def compute_landauer_cost(
    S: float,
    eps: float,
    k_b: float = 1.38e-23,
    T_env: float = 310.0,
    kappa_meta: float = 1.0
) -> float:
    """Compute Landauer cost.
    
    E_min = κ_meta · N_erase · k_B · T_env · ln(2)
    where N_erase ≈ log₂(S / ε_stab)
    
    Args:
        S: Signal magnitude
        eps: Stability constant
        k_b: Boltzmann constant (default 1.38e-23 J/K)
        T_env: Environment temperature (default 310 K)
        kappa_meta: Metabolic efficiency (default 1.0)
        
    Returns:
        Thermodynamic cost in Joules
        
    Spec Reference: §11 Thermodynamic Constraints
    
    Example:
        >>> cost = compute_landauer_cost(S=0.5, eps=1e-8)
        >>> print(f"Thermodynamic cost: {cost:.2e} J")
    """
```

---

## Observable Mapping

### `NeuralObservableExtractor`

Extract neural observables from APGI dynamics.

```python
class NeuralObservableExtractor:
    """Extract neural observables per spec §14.
    
    Maps:
    - S(t) → Gamma-band power (30-100 Hz)
    - θ(t) → P300/N200 ERP amplitude
    - B(t) → Global ignition (gamma synchrony)
    """
    
    def step(self, S: float, theta: float, B: int) -> dict:
        """Extract neural observables.
        
        Args:
            S: Signal magnitude
            theta: Threshold
            B: Ignition binary
            
        Returns:
            Dictionary with:
            - gamma_power: Gamma-band power
            - erp_amplitude: ERP amplitude
            - ignition_rate: Ignition rate
        """
    
    def extract_gamma_power(self) -> float:
        """Get gamma-band power from history."""
    
    def extract_erp_amplitude(self) -> float:
        """Get ERP amplitude from history."""
```

### `BehavioralObservableExtractor`

Extract behavioral observables.

```python
class BehavioralObservableExtractor:
    """Extract behavioral observables per spec §14.
    
    Maps:
    - S(t) → Perceptual sensitivity (d')
    - θ(t) → RT variability, response criterion
    - B(t) → Overt decision/button press
    """
    
    def step(self, S: float, theta: float, B: int) -> dict:
        """Extract behavioral observables.
        
        Args:
            S: Signal magnitude
            theta: Threshold
            B: Ignition binary
            
        Returns:
            Dictionary with:
            - rt_variability: RT variability
            - response_criterion: Response criterion
            - decision_rate: Decision rate
        """
```

---

## Stability Analysis

### `StabilityAnalyzer`

Analyze system stability.

```python
class StabilityAnalyzer:
    """Stability analysis per spec §7.
    
    Computes Jacobian, eigenvalues, and stability.
    """
    
    def __init__(self, config: dict):
        """Initialize stability analyzer.
        
        Args:
            config: Configuration dictionary
        """
    
    def check_stability(self) -> dict:
        """Check fixed-point stability.
        
        Returns:
            Dictionary with:
            - stable: Boolean stability flag
            - eigenvalues: Eigenvalue array
            - max_eigenvalue: Maximum eigenvalue magnitude
            - jacobian: Jacobian matrix
        """
    
    def analyze_bifurcation(
        self,
        param_name: str,
        param_range: np.ndarray
    ) -> dict:
        """Analyze bifurcation in parameter.
        
        Args:
            param_name: Parameter name
            param_range: Parameter values to test
            
        Returns:
            Bifurcation analysis results
        """
```

---

## Validation

### `validate_config`

Validate configuration against spec constraints.

```python
def validate_config(config: dict) -> None:
    """Validate configuration.
    
    Checks all 8 constraint categories:
    1. Neuromodulator separation
    2. Signal accumulation
    3. Threshold dynamics
    4. Ignition dynamics
    5. Continuous-time SDE
    6. Hierarchical parameters
    7. Precision parameters
    8. Numerical stability
    
    Args:
        config: Configuration dictionary
        
    Raises:
        ValueError: If any constraint violated
        
    Spec Reference: §15 Design Constraints
    
    Example:
        >>> from config import CONFIG
        >>> validate_config(CONFIG)
        >>> print("Configuration valid!")
    """
```

---

## Configuration

### `CONFIG`

Default configuration dictionary.

```python
CONFIG = {
    # Signal preprocessing (§1)
    "tau_s": 5.0,  # Signal timescale (ms)
    "tau_pi": 1000.0,  # Precision timescale (ms)
    "tau_theta": 1000.0,  # Threshold timescale (ms)
    "eps_stab": 1e-8,  # Stability constant
    
    # Precision system (§2)
    "pi_min": 0.01,  # Minimum precision
    "pi_max": 100.0,  # Maximum precision
    "ne_on_precision": False,  # NE modulates precision
    "ne_on_threshold": True,  # NE modulates threshold
    
    # Signal accumulation (§3)
    "lam": 0.2,  # Integration rate
    "use_log_compression": False,  # Log-compress signal
    
    # Threshold dynamics (§4)
    "c1": 0.2,  # Signal cost coefficient
    "c2": 0.1,  # Ignition cost coefficient
    "eta": 0.1,  # Learning rate
    "delta_reset": 0.5,  # Refractory boost
    "kappa": 0.15,  # Decay rate
    "theta_base": 0.5,  # Baseline threshold
    
    # Ignition mechanism (§5)
    "ignite_tau": 0.5,  # Sigmoid temperature
    "use_soft_ignition": False,  # Soft vs hard ignition
    
    # Post-ignition reset (§6)
    "reset_factor": 0.5,  # Signal reset factor
    
    # Continuous-time SDE (§7)
    "dt": 1.0,  # Time step (ms)
    "sigma_s": 0.1,  # Signal noise amplitude
    
    # Hierarchical architecture (§8)
    "use_hierarchy": False,  # Enable hierarchical system
    "n_levels": 5,  # Number of hierarchical levels
    "timescale_k": 1.6,  # Timescale ratio
    
    # Oscillatory coupling (§9)
    "use_kuramoto": False,  # Enable Kuramoto oscillators
    "kuramoto_tau_xi": 1.0,  # OU noise timescale
    "kuramoto_sigma_xi": 0.1,  # OU noise amplitude
    
    # Reservoir layer (§10)
    "use_reservoir": False,  # Enable reservoir layer
    "reservoir_size": 100,  # Reservoir neuron count
    "reservoir_tau": 1.0,  # Reservoir timescale
    
    # Thermodynamic constraints (§11)
    "use_thermodynamics": False,  # Enable Landauer cost
    "k_boltzmann": 1.38e-23,  # Boltzmann constant (J/K)
    "T_env": 310.0,  # Environment temperature (K)
    "kappa_meta": 1.0,  # Metabolic efficiency
    
    # Observable mapping (§14)
    "use_observable_mapping": False,  # Enable observable extraction
    
    # Stability analysis (§7)
    "use_stability_analysis": False,  # Enable stability analysis
}
```

---

## Usage Examples

### Basic Usage

```python
from pipeline import APGIPipeline
from config import CONFIG

# Initialize pipeline
pipeline = APGIPipeline(CONFIG)

# Run single step
result = pipeline.step(
    x_e=0.5,      # Exteroceptive signal
    x_hat_e=0.3,  # Exteroceptive prediction
    x_i=0.2,      # Interoceptive signal
    x_hat_i=0.1   # Interoceptive prediction
)

# Access results
print(f"Signal: {result['S']:.4f}")
print(f"Threshold: {result['theta']:.4f}")
print(f"Ignition: {result['B']}")
```

### Advanced Usage with Observable Mapping

```python
config = CONFIG.copy()
config["use_observable_mapping"] = True
config["use_kuramoto"] = True

pipeline = APGIPipeline(config)

# Run simulation
for t in range(1000):
    result = pipeline.step(x_e=0.5, x_hat_e=0.3, x_i=0.2, x_hat_i=0.1)
    
    # Access neural observables
    if "neural_gamma_power" in result:
        print(f"Gamma power: {result['neural_gamma_power']:.4f}")
    
    # Access behavioral observables
    if "behavioral_rt_variability" in result:
        print(f"RT variability: {result['behavioral_rt_variability']:.4f}")
```

---

## Error Handling

All functions raise `ValueError` with descriptive messages when constraints are violated. Example:

```python
try:
    pipeline = APGIPipeline(invalid_config)
except ValueError as e:
    print(f"Configuration error: {e}")
    print("See docs/PARAMETER_CONSTRAINTS.md for valid ranges")
```

---
