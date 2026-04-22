"""Comprehensive parameter validation for APGI configuration.

APGI Specification §15: Design Constraints

This module enforces all parameter constraints specified in the APGI specification
to prevent invalid configurations and ensure numerical stability.

Constraints enforced:
    - Neuromodulator separation (§2.3-2.4)
    - Parameter ranges (§15)
    - Numerical stability bounds
    - Integration step size constraints (§7.4)
    - Timescale hierarchy constraints (§8.1)
    - Precision bounds (§2.2)

All constraints include spec section references for traceability.
"""

from __future__ import annotations

from typing import Any


class ValidationError(ValueError):
    """Raised when configuration validation fails."""

    pass


def validate_config(config: dict) -> None:
    """Validate APGI configuration against all spec constraints.

    Performs comprehensive validation of all parameters. Raises ValidationError
    if any constraint is violated.

    Args:
        config: Configuration dictionary

    Raises:
        ValidationError: If any constraint is violated

    Examples:
        >>> config = {"lam": 0.2, "kappa": 0.15, "dt": 1.0}
        >>> validate_config(config)  # Passes

        >>> config = {"lam": 1.5}  # Invalid
        >>> validate_config(config)  # Raises ValidationError
    """
    # Validate each constraint category
    _validate_neuromodulator_separation(config)
    _validate_signal_accumulation(config)
    _validate_threshold_dynamics(config)
    _validate_ignition_dynamics(config)
    _validate_continuous_time_sde(config)
    _validate_hierarchical_parameters(config)
    _validate_precision_parameters(config)
    _validate_numerical_stability(config)


def _validate_neuromodulator_separation(config: dict) -> None:
    """Validate neuromodulator separation constraints (§2.3-2.4).

    Spec §2.3-2.4: NE cannot modulate both precision and threshold simultaneously.
    This prevents double-counting of norepinephrine effects.
    """
    ne_on_precision = config.get("ne_on_precision", False)
    ne_on_threshold = config.get("ne_on_threshold", False)

    if ne_on_precision and ne_on_threshold:
        raise ValidationError(
            "NE cannot modulate both precision and threshold. "
            "Set exactly one to True. Spec §2.3-2.4."
        )


def _validate_signal_accumulation(config: dict) -> None:
    """Validate signal accumulation constraints (§3.2).

    Spec §3.2: Integration rate λ must be in (0, 1) for leaky integrator.
    """
    lam = config.get("lam", 0.2)

    if not (0 < lam < 1):
        raise ValidationError(
            f"lam must be in (0, 1), got {lam}. "
            "Spec §3.2: S(t+1) = (1-λ)S(t) + λ·S_inst(t)"
        )


def validate_reset_factor(reset_factor: float) -> None:
    """Validate reset factor ρ per APGI spec §6.

    Spec requirement: ρ ∈ (0, 1)
    - ρ close to 0: Full reset (strong refractory period)
    - ρ close to 1: Partial reset (mild inhibition)

    Args:
        reset_factor: Signal reset factor

    Raises:
        ValidationError: If ρ not in (0, 1)
    """
    if not (0.0 < reset_factor < 1.0):
        raise ValidationError(
            f"reset_factor must be in (0, 1), got {reset_factor}. "
            "Spec §6: Signal reset S ← ρ·S requires ρ ∈ (0, 1). "
            "ρ ≤ 0 causes signal to grow; ρ ≥ 1 prevents reset."
        )


def _validate_threshold_dynamics(config: dict) -> None:
    """Validate threshold dynamics constraints (§4, §6).

    Spec §4.5: Exponential decay rate κ must be > 0
    Spec §6.1: Reset factor ρ must be in (0, 1)
    """
    kappa = config.get("kappa", 0.15)

    if kappa <= 0:
        raise ValidationError(
            f"kappa must be > 0, got {kappa}. "
            "Spec §4.5: θ(t+1) = θ_base + (θ(t) - θ_base)·exp(-κ)"
        )

    # Reset factor ρ (if exposed)
    if "reset_factor" in config:
        validate_reset_factor(config["reset_factor"])


def _validate_ignition_dynamics(config: dict) -> None:
    """Validate ignition mechanism constraints (§5.2).

    Spec §5.2: Sigmoid temperature τ_σ must be > 0
    """
    tau_sigma = config.get("ignite_tau", 0.5)

    if tau_sigma <= 0:
        raise ValidationError(
            f"ignite_tau (τ_σ) must be > 0, got {tau_sigma}. "
            "Spec §5.2: P_ign = σ([S-θ]/τ_σ)"
        )


def _validate_continuous_time_sde(config: dict) -> None:
    """Validate continuous-time SDE constraints (§7.4).

    Spec §7.4: Euler-Maruyama integration requires:
        dt ≤ min(τ_S, τ_θ, τ_Π) / 10

    This ensures numerical stability of the SDE integration.
    """
    dt = config.get("dt", 1.0)
    tau_s = config.get("tau_s", 5.0)
    tau_theta = config.get("tau_theta", 1000.0)
    tau_pi = config.get("tau_pi", 1000.0)

    if dt <= 0:
        raise ValidationError(
            f"dt must be > 0, got {dt}. " "Spec §7.4: Euler-Maruyama integration step"
        )

    min_tau = min(tau_s, tau_theta, tau_pi)
    max_dt = min_tau / 10.0

    if dt > max_dt:
        raise ValidationError(
            f"dt={dt} exceeds max {max_dt:.4f}. "
            f"Spec §7.4 requires dt ≤ min(τ_S, τ_θ, τ_Π) / 10. "
            f"(τ_S={tau_s}, τ_θ={tau_theta}, τ_Π={tau_pi})"
        )


def _validate_hierarchical_parameters(config: dict) -> None:
    """Validate hierarchical architecture constraints (§8.1).

    Spec §8.1: Timescale ratio k must be > 1
    Spec §8.2: All timescales τ_ℓ must be > 1
    """
    if config.get("use_hierarchical", False):
        k = config.get("timescale_k", 1.6)

        if k <= 1:
            raise ValidationError(
                f"timescale_k must be > 1, got {k}. "
                "Spec §8.1: τ_ℓ = τ_0·k^ℓ requires k > 1"
            )

        # Validate that all timescales are > 1
        tau_0 = config.get("tau_0", 1.0)
        n_levels = config.get("n_levels", 3)

        for level in range(n_levels):
            tau_ell = tau_0 * (k**level)
            if tau_ell <= 1:
                raise ValidationError(
                    f"τ_{level} = {tau_ell:.2f} ≤ 1. "
                    "Spec §8.2: All timescales must be > 1"
                )


def _validate_precision_parameters(config: dict) -> None:
    """Validate precision system constraints (§2.2).

    Spec §2.2: Precision bounds Π_min < Π_max
    Spec §2.2: Clamping prevents numerical issues
    """
    pi_min = config.get("pi_min", 1e-4)
    pi_max = config.get("pi_max", 1e4)

    if pi_min <= 0:
        raise ValidationError(
            f"pi_min must be > 0, got {pi_min}. " "Spec §2.2: Precision lower bound"
        )

    if pi_max <= pi_min:
        raise ValidationError(
            f"pi_max must be > pi_min, got pi_max={pi_max}, pi_min={pi_min}. "
            "Spec §2.2: Precision bounds"
        )

    # Reasonable range check
    if pi_max / pi_min > 1e8:
        import warnings

        warnings.warn(
            f"Precision range very large: pi_max/pi_min = {pi_max / pi_min:.1e}. "
            "May cause numerical issues. Spec §2.2 recommends (0.01, 100).",
            RuntimeWarning,
            stacklevel=3,
        )


def _validate_numerical_stability(config: dict) -> None:
    """Validate general numerical stability constraints.

    Checks for reasonable parameter ranges and potential numerical issues.
    """
    eps = config.get("eps", 1e-8)

    if eps <= 0 or eps >= 1:
        raise ValidationError(
            f"eps must be in (0, 1), got {eps}. " "Numerical stability threshold"
        )

    # Check for reasonable learning rates
    eta = config.get("eta", 0.1)
    if eta <= 0 or eta > 1:
        raise ValidationError(
            f"eta must be in (0, 1], got {eta}. " "Spec §4.1: Threshold learning rate"
        )

    # Check for reasonable noise levels
    noise_std = config.get("noise_std", 0.01)
    if noise_std < 0:
        raise ValidationError(
            f"noise_std must be ≥ 0, got {noise_std}. "
            "Spec §7.2: SDE diffusion coefficient"
        )


def validate_parameter(
    name: str, value: Any, constraint: str, spec_section: str = ""
) -> None:
    """Validate a single parameter against a constraint.

    Args:
        name: Parameter name
        value: Parameter value
        constraint: Constraint description (e.g., "> 0", "in (0, 1)")
        spec_section: Spec section reference (e.g., "§3.2")

    Raises:
        ValidationError: If constraint violated

    Examples:
        >>> validate_parameter("lam", 0.2, "in (0, 1)", "§3.2")
        >>> validate_parameter("lam", 1.5, "in (0, 1)", "§3.2")  # Raises
    """
    # Parse constraint
    if constraint.startswith("in ("):
        # Parse "in (a, b)" format
        parts = constraint[4:-1].split(",")
        a, b = float(parts[0].strip()), float(parts[1].strip())
        if not (a < value < b):
            raise ValidationError(
                f"{name}={value} not {constraint}. Spec {spec_section}"
            )

    elif constraint.startswith(">"):
        # Parse "> a" format
        threshold = float(constraint[1:].strip())
        if value <= threshold:
            raise ValidationError(
                f"{name}={value} not {constraint}. Spec {spec_section}"
            )

    elif constraint.startswith(">="):
        # Parse ">= a" format
        threshold = float(constraint[2:].strip())
        if value < threshold:
            raise ValidationError(
                f"{name}={value} not {constraint}. Spec {spec_section}"
            )

    elif constraint.startswith("<"):
        # Parse "< a" format
        threshold = float(constraint[1:].strip())
        if value >= threshold:
            raise ValidationError(
                f"{name}={value} not {constraint}. Spec {spec_section}"
            )

    elif constraint.startswith("<="):
        # Parse "<= a" format
        threshold = float(constraint[2:].strip())
        if value > threshold:
            raise ValidationError(
                f"{name}={value} not {constraint}. Spec {spec_section}"
            )


def get_constraint_summary() -> dict:
    """Get summary of all constraints for documentation.

    Returns:
        Dictionary mapping constraint categories to lists of constraints

    Examples:
        >>> summary = get_constraint_summary()
        >>> print(summary["Signal Accumulation"])
        ['lam in (0, 1)']
    """
    return {
        "Neuromodulator Separation": [
            "NE cannot modulate both precision and threshold (§2.3-2.4)",
        ],
        "Signal Accumulation": [
            "lam in (0, 1) (§3.2)",
        ],
        "Threshold Dynamics": [
            "kappa > 0 (§4.5)",
            "reset_factor in (0, 1) (§6.1)",
        ],
        "Ignition Mechanism": [
            "ignite_tau > 0 (§5.2)",
        ],
        "Continuous-Time SDE": [
            "dt > 0 (§7.4)",
            "dt ≤ min(τ_S, τ_θ, τ_Π) / 10 (§7.4)",
        ],
        "Hierarchical Architecture": [
            "timescale_k > 1 (§8.1)",
            "All τ_ℓ > 1 (§8.2)",
        ],
        "Precision System": [
            "pi_min > 0 (§2.2)",
            "pi_max > pi_min (§2.2)",
        ],
        "Numerical Stability": [
            "eps in (0, 1)",
            "eta in (0, 1]",
            "noise_std ≥ 0",
        ],
    }


def print_constraint_summary() -> None:
    """Print human-readable summary of all constraints."""
    summary = get_constraint_summary()
    print("\n" + "=" * 70)
    print("APGI PARAMETER CONSTRAINTS (Spec §15)")
    print("=" * 70)

    for category, constraints in summary.items():
        print(f"\n{category}:")
        for constraint in constraints:
            print(f"  • {constraint}")

    print("\n" + "=" * 70)
