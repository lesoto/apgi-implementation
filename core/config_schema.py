"""Pydantic configuration schema for type-safe APGI configuration.

Provides strict validation with clear error messages for production deployments.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator


class APGIConfig(BaseModel):
    """Production-grade APGI configuration with strict validation.

    All parameters validated at initialization with clear error messages.
    No auto-adjustments - invalid configurations raise ValidationError.
    """

    # Initial states
    S0: float = Field(default=0.0, description="Initial signal value")
    theta_0: float = Field(default=1.0, gt=0, description="Initial threshold")
    theta_base: float = Field(default=1.0, gt=0, description="Base threshold")
    sigma2_e0: float = Field(default=1.0, gt=0, description="Initial exteroceptive variance")
    sigma2_i0: float = Field(default=1.0, gt=0, description="Initial interoceptive variance")

    # Numerical stability
    eps: float = Field(default=1e-8, gt=0, lt=1, description="Numerical stability threshold")
    pi_min: float = Field(default=0.01, gt=0, description="Minimum precision")
    pi_max: float = Field(default=100.0, gt=0, description="Maximum precision")

    # EMA variance update
    alpha_e: float = Field(default=0.05, gt=0, lt=1, description="Exteroceptive EMA rate")
    alpha_i: float = Field(default=0.05, gt=0, lt=1, description="Interoceptive EMA rate")

    # Variance estimation method
    variance_method: Literal["ema", "sliding_window"] = Field(
        default="ema", description="Variance estimation method"
    )
    T_win: int = Field(default=50, gt=0, description="Sliding window size")

    # Neuromodulation
    g_ach: float = Field(default=1.0, ge=0, description="Acetylcholine gain")
    g_ne: float = Field(default=1.0, ge=0, description="Norepinephrine gain")

    # Dopaminergic bias
    beta: float = Field(default=1.15, ge=0, description="Dopaminergic bias")
    beta_da: float | None = Field(
        default=None, description="Alias for beta (backward compatibility)"
    )

    # NE double counting prevention
    ne_on_precision: bool = Field(default=True, description="NE modulates precision")
    ne_on_threshold: bool = Field(default=False, description="NE modulates threshold")
    gamma_ne: float = Field(default=0.1, ge=0, le=1, description="NE modulation strength")
    kappa: float = Field(default=0.15, gt=0, le=1, description="Threshold decay rate")

    # Signal accumulation
    lam: float = Field(default=0.2, gt=0, lt=1, description="Signal integration rate")
    signal_log_nonlinearity: bool = Field(default=True, description="Enable log stabilization")
    use_canonical_discrete_mode: bool = Field(
        default=False, description="Use discrete accumulation instead of ODE"
    )

    # Threshold update + refractory dynamics
    eta: float = Field(default=0.1, gt=0, le=1, description="Threshold learning rate")
    delta: float = Field(default=0.5, ge=0, description="Refractory boost")

    # Post-ignition signal reset
    reset_factor: float = Field(default=0.1, gt=0, lt=1, description="Signal reset factor (rho)")

    # Threshold adaptation timescales
    tau_theta: float = Field(
        default=20.0, gt=0, description="Threshold adaptation timescale (allostatic)"
    )
    tau_theta_recovery: float = Field(
        default=0.45, gt=0, description="Threshold recovery timescale (perceptual)"
    )

    # Hierarchical cascade tuning
    KAPPA_UP: float = Field(default=0.1, ge=0, le=1, description="Bottom-up cascade strength")
    KAPPA_DOWN: float = Field(default=0.1, ge=0, le=1, description="Top-down coupling strength")

    # Cost-value model
    use_realistic_cost: bool = Field(default=True, description="Use realistic metabolic cost")
    c0: float = Field(default=0.0, ge=0, description="Base metabolic cost")
    c1: float = Field(default=0.2, ge=0, description="Signal-dependent cost coefficient")
    c2: float = Field(default=0.5, ge=0, description="Ignition cost coefficient")
    v1: float = Field(default=0.5, ge=0, description="Exteroceptive value weight")
    v2: float = Field(default=0.5, ge=0, description="Interoceptive value weight")

    # Ignition dynamics
    ignite_tau: float = Field(default=0.5, gt=0, description="Ignition sigmoid temperature")
    tau_sigma: float | None = Field(
        default=None, description="Alias for ignite_tau (backward compatibility)"
    )
    stochastic_ignition: bool = Field(default=False, description="Enable stochastic ignition")

    # Continuous-time parameters
    tau_s: float = Field(default=5.0, gt=0, description="Signal decay time constant")
    dt: float = Field(default=0.5, gt=0, description="Integration time step")
    noise_std: float = Field(default=0.01, ge=0, description="SDE diffusion coefficient")

    # Generative model dynamics
    use_internal_predictions: bool = Field(default=True, description="Enable internal predictions")
    kappa_e: float = Field(default=0.01, ge=0, description="Exteroceptive prediction rate")
    kappa_i: float = Field(default=0.01, ge=0, description="Interoceptive prediction rate")

    # Multi-scale parameters
    timescale_k: float = Field(default=1.6, gt=1, lt=3, description="Timescale expansion factor")

    # Thermodynamic constraints
    use_thermodynamic_cost: bool = Field(default=False, description="Enable Landauer cost")
    k_boltzmann: float = Field(default=1.38e-23, gt=0, description="Boltzmann constant")
    T_env: float = Field(default=310.0, gt=0, description="Environmental temperature (K)")
    kappa_meta: float = Field(default=1.0, gt=0, description="Metabolic efficiency factor")
    kappa_units: Literal["dimensionless", "joules_per_bit"] = Field(
        default="dimensionless", description="Metabolic cost units"
    )

    # BOLD calibration
    use_bold_calibration: bool = Field(default=False, description="Enable BOLD calibration")
    bold_conversion_factor: float = Field(
        default=1.2e-18, gt=0, description="Joules per 1% BOLD change per cm³"
    )
    bold_tissue_volume: float = Field(default=1.0, gt=0, description="Tissue volume in cm³")
    bold_ignition_spike_factor: float = Field(
        default=1.075, gt=0, description="Energy spike factor during ignition"
    )

    # Reservoir layer
    use_reservoir: bool = Field(default=False, description="Enable reservoir computing")
    reservoir_size: int = Field(default=100, gt=0, description="Reservoir units")
    reservoir_tau: float = Field(default=1.0, gt=0, description="Reservoir time constant")
    reservoir_spectral_radius: float = Field(
        default=0.9, gt=0, lt=1, description="Reservoir spectral radius"
    )
    reservoir_input_scale: float = Field(default=0.1, gt=0, description="Input scaling")
    reservoir_readout_method: Literal["linear", "energy"] = Field(
        default="linear", description="Reservoir readout method"
    )
    reservoir_amplification: float = Field(
        default=0.0, ge=0, description="Suprathreshold amplification strength"
    )

    # Kuramoto oscillators
    use_kuramoto: bool = Field(default=False, description="Enable Kuramoto oscillators")
    kuramoto_tau_xi: float = Field(default=1.0, gt=0, description="OU noise correlation timescale")
    kuramoto_sigma_xi: float = Field(default=0.1, ge=0, description="OU noise amplitude")
    kuramoto_reset_amount: float = Field(
        default=3.14159, gt=0, description="Phase reset on ignition (radians)"
    )

    # Hierarchical mode
    hierarchical_mode: Literal["off", "basic", "advanced", "full"] | None = Field(
        default=None, description="Hierarchical mode preset"
    )
    use_hierarchical: bool = Field(default=False, description="Enable hierarchical processing")
    use_hierarchical_precision_ode: bool = Field(default=False, description="Enable precision ODE")
    use_phase_modulation: bool = Field(default=False, description="Enable phase modulation")
    n_levels: int = Field(default=3, ge=1, le=10, description="Hierarchy levels")

    # Observable mapping
    use_observable_mapping: bool = Field(
        default=False, description="Enable neural/behavioral observable extraction"
    )

    # Stability analysis
    use_stability_analysis: bool = Field(default=False, description="Enable stability analysis")

    @field_validator("pi_max")
    @classmethod
    def validate_pi_range(cls, v: float, info: ValidationInfo) -> float:
        """Ensure pi_max > pi_min."""
        if info.data.get("pi_min") is not None and v <= info.data["pi_min"]:
            raise ValueError("pi_max must be greater than pi_min")
        return v

    @model_validator(mode="after")
    def validate_ne_separation(self) -> "APGIConfig":
        """Prevent NE double-counting (Spec §2.3-2.4)."""
        if self.ne_on_precision and self.ne_on_threshold:
            raise ValueError(
                "NE cannot modulate both precision and threshold. "
                "Set exactly one to True. Spec §2.3-2.4."
            )
        return self

    @model_validator(mode="after")
    def validate_ne_threshold_stability(self) -> "APGIConfig":
        """Validate NE threshold mode parameters for stability."""
        if self.ne_on_threshold and self.gamma_ne >= 0.1:
            raise ValueError(
                f"ne_on_threshold=True with gamma_ne={self.gamma_ne} causes threshold instability. "
                "Use gamma_ne <= 0.01 or kappa >= 0.15. Spec §4.4."
            )
        return self

    @model_validator(mode="after")
    def validate_dt_stability(self) -> "APGIConfig":
        """Validate integration step size (Spec §7.4)."""
        min_tau = min(self.tau_s, self.tau_theta, 1000.0)  # Default tau_pi = 1000
        max_dt = min_tau / 10.0
        if self.dt > max_dt:
            raise ValueError(
                f"dt={self.dt} exceeds max {max_dt:.4f}. "
                f"Spec §7.4 requires dt <= min(τ_S, τ_θ, τ_Π) / 10."
            )
        return self

    @model_validator(mode="after")
    def validate_learning_rates(self) -> "APGIConfig":
        """Validate generative model learning rates (Spec §1.4)."""
        if self.use_internal_predictions:
            max_kappa = 2.0 / self.pi_max
            if self.kappa_e >= max_kappa:
                raise ValueError(
                    f"kappa_e={self.kappa_e} >= {max_kappa:.4f}. "
                    f"Spec §1.4 requires κ_e < 2/Π_max for stability."
                )
            if self.kappa_i >= max_kappa:
                raise ValueError(
                    f"kappa_i={self.kappa_i} >= {max_kappa:.4f}. "
                    f"Spec §1.4 requires κ_i < 2/Π_max for stability."
                )
        return self

    @model_validator(mode="after")
    def apply_backward_compat(self) -> "APGIConfig":
        """Apply backward compatibility aliases and ensure consistency."""
        # 1. Somatic bias (beta/beta_da)
        if self.beta_da is not None:
            # beta_da always takes precedence if provided
            if abs(self.beta_da - self.beta) > 0.01 or self.beta == 1.15:
                self.beta = self.beta_da

        # 2. Ignition temperature (tau_sigma/ignite_tau)
        if self.tau_sigma is not None:
            # tau_sigma always takes precedence if provided
            if abs(self.tau_sigma - self.ignite_tau) > 0.01 or self.ignite_tau == 0.5:
                self.ignite_tau = self.tau_sigma

        return self

    def to_dict(self) -> dict:
        """Convert to dictionary for legacy compatibility."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict) -> "APGIConfig":
        """Create from dictionary."""
        return cls(**data)


def create_production_config(strict: bool = True, **overrides: Any) -> APGIConfig | dict:
    """Create a production-ready configuration.

    Args:
        strict: If True, return APGIConfig (strict validation). If False, return dict.
        **overrides: Configuration overrides

    Returns:
        APGIConfig instance or dictionary based on strict mode
    """
    config = APGIConfig(**overrides)
    if strict:
        return config
    return config.to_dict()
