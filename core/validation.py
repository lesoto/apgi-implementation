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

import warnings
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.logging_config import get_logger

    logger = get_logger("apgi.validation")


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
    _validate_learning_rates(config)  # Spec §1.4
    _validate_ema_parameters(config)  # Spec §1.2
    _validate_sliding_window_params(config)  # Spec §1.2
    _validate_numerical_stability(config)

    # Validate post-ignition reset factor (§6)
    if "reset_factor" in config:
        validate_reset_factor(config["reset_factor"])

    # Validate neuromodulatory gains
    g_ach = config.get("g_ach", 1.0)
    g_ne = config.get("g_ne", 1.0)
    if g_ach < 0:
        raise ValidationError(f"g_ach={g_ach} must be ≥ 0. Spec §2.3: ACh gain")
    if g_ne < 0:
        raise ValidationError(f"g_ne={g_ne} must be ≥ 0. Spec §2.3: NE gain")

    # Validate dopaminergic bias
    beta = config.get("beta", 1.15)
    if beta < 0:
        raise ValidationError(f"beta={beta} must be ≥ 0. Spec §2.4: DA bias")


def _validate_neuromodulator_separation(config: dict) -> None:
    """Validate neuromodulator separation constraints (§2.3-2.4).

    Spec §2.3-2.4: NE cannot modulate both precision and threshold simultaneously.
    This prevents double-counting of norepinephrine effects.
    """
    ne_on_precision = config.get("ne_on_precision", False)
    ne_on_threshold = config.get("ne_on_threshold", False)

    if ne_on_precision and ne_on_threshold:
        raise ValidationError(
            "NE cannot modulate both precision and threshold (double-counts). "
            "Set exactly one to True. Spec §2.3-2.4."
        )


def _validate_signal_accumulation(config: dict) -> None:
    """Validate signal accumulation constraints (§3.2).

    Spec §3.2: Integration rate λ must be in (0, 1) for leaky integrator.
    """
    lam = config.get("lam", 0.2)

    if not (0 < lam < 1):
        raise ValidationError(
            f"lam must be in (0, 1), got {lam}. " "Spec §3.2: S(t+1) = (1-λ)S(t) + λ·S_inst(t)"
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
    Spec §6.1: Reset factor ρ must be in (0, 1) - validated at step() time
    """
    kappa = config.get("kappa", 0.15)

    if kappa <= 0:
        raise ValidationError(
            f"kappa must be > 0, got {kappa}. "
            "Spec §4.5: θ(t+1) = θ_base + (θ(t) - θ_base)·exp(-κ)"
        )


def _validate_ignition_dynamics(config: dict) -> None:
    """Validate ignition mechanism constraints (§5.2).

    Spec §5.2: Sigmoid temperature τ_σ must be > 0
    """
    tau_sigma = config.get("ignite_tau", 0.5)

    if tau_sigma <= 0:
        raise ValidationError(
            f"ignite_tau (τ_σ) must be > 0, got {tau_sigma}. " "Spec §5.2: P_ign = σ([S-θ]/τ_σ)"
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
                f"timescale_k must be > 1, got {k}. " "Spec §8.1: τ_ℓ = τ_0·k^ℓ requires k > 1"
            )

        # Validate that all timescales are > 1
        tau_0 = config.get("tau_0", 1.0)
        n_levels = config.get("n_levels", 3)

        for level in range(n_levels):
            tau_ell = tau_0 * (k**level)
            if tau_ell <= 1:
                raise ValidationError(
                    f"τ_{level} = {tau_ell:.2f} ≤ 1. " "Spec §8.2: All timescales must be > 1"
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

    # Reasonable range check - warn in production
    if pi_max / pi_min > 1e8:
        warnings.warn(
            f"Precision range very large: pi_max/pi_min = {pi_max / pi_min:.1e}. "
            "May cause numerical issues. Spec §2.2 recommends (0.01, 100). "
            "Use strict_mode=True to enforce strict validation.",
            RuntimeWarning,
            stacklevel=2,
        )


def _validate_learning_rates(config: dict) -> None:
    """Validate generative model learning rates (Spec §1.4).

    Spec §1.4: Learning rates κ_e, κ_i must satisfy κ < 2/Π_max
    for gradient descent convergence in the generative model.
    """
    if not config.get("use_internal_predictions", False):
        return  # Skip if generative model disabled

    kappa_e = config.get("kappa_e", 0.01)
    kappa_i = config.get("kappa_i", 0.01)
    pi_max = config.get("pi_max", 1e4)

    max_kappa = 2.0 / pi_max

    if kappa_e >= max_kappa:
        raise ValidationError(
            f"kappa_e={kappa_e} >= {max_kappa:.4f}. "
            f"Spec §1.4 requires κ_e < 2/Π_max for generative model stability. "
            f"Current Π_max={pi_max}, max allowed κ_e={max_kappa:.4f}"
        )

    if kappa_i >= max_kappa:
        raise ValidationError(
            f"kappa_i={kappa_i} >= {max_kappa:.4f}. "
            f"Spec §1.4 requires κ_i < 2/Π_max for generative model stability. "
            f"Current Π_max={pi_max}, max allowed κ_i={max_kappa:.4f}"
        )


def _validate_ema_parameters(config: dict) -> None:
    """Validate EMA variance estimation parameters (Spec §1.2).

    Spec §1.2: EMA smoothing parameters α must be in (0, 1].
    """
    variance_method = config.get("variance_method", "ema")

    if variance_method == "ema":
        alpha_e = config.get("alpha_e", 0.05)
        alpha_i = config.get("alpha_i", 0.05)

        if not (0 < alpha_e <= 1):
            raise ValidationError(f"alpha_e={alpha_e} must be in (0, 1] for EMA. Spec §1.2")

        if not (0 < alpha_i <= 1):
            raise ValidationError(f"alpha_i={alpha_i} must be in (0, 1] for EMA. Spec §1.2")


def _validate_sliding_window_params(config: dict) -> None:
    """Validate sliding window variance estimation parameters.

    T_win must be positive integer for sliding window method.
    """
    variance_method = config.get("variance_method", "ema")

    if variance_method == "sliding_window":
        T_win = config.get("T_win", 50)

        if not isinstance(T_win, int) or T_win <= 0:
            raise ValidationError(
                f"T_win={T_win} must be positive integer for sliding window. "
                f"Spec §1.2 (Method B)"
            )

        # Warn if window is too small for meaningful Bessel correction
        if T_win < 5:
            warnings.warn(
                f"T_win={T_win} is very small. Bessel correction may be unstable. "
                f"Recommend T_win >= 10. Spec §1.2",
                RuntimeWarning,
                stacklevel=2,
            )


def _validate_numerical_stability(config: dict) -> None:
    """Validate general numerical stability constraints.

    Checks for reasonable parameter ranges and potential numerical issues.
    """
    eps = config.get("eps", 1e-8)

    if eps <= 0 or eps >= 1:
        raise ValidationError(f"eps must be in (0, 1), got {eps}. " "Numerical stability threshold")

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
            f"noise_std must be ≥ 0, got {noise_std}. " "Spec §7.2: SDE diffusion coefficient"
        )

    # Validate cost-value coefficients
    c0 = config.get("c0", 0.0)
    c1 = config.get("c1", 0.2)
    c2 = config.get("c2", 0.5)
    v1 = config.get("v1", 0.5)
    v2 = config.get("v2", 0.5)

    for name, value in [("c0", c0), ("c1", c1), ("c2", c2), ("v1", v1), ("v2", v2)]:
        if value < 0:
            raise ValidationError(f"{name}={value} must be ≥ 0. Spec §4.2-4.3: Cost-value model")

    # Validate refractory boost
    delta = config.get("delta", 0.5)
    if delta < 0:
        raise ValidationError(f"delta={delta} must be ≥ 0. Spec §6: Refractory boost")

    # Validate log nonlinearity toggle
    signal_log_nonlinearity = config.get("signal_log_nonlinearity", True)
    if not isinstance(signal_log_nonlinearity, bool):
        raise ValidationError(
            f"signal_log_nonlinearity must be boolean, got {type(signal_log_nonlinearity)}. "
            f"Spec §3.3: Optional log compression"
        )


def validate_parameter(name: str, value: Any, constraint: str, spec_section: str = "") -> None:
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
            raise ValidationError(f"{name}={value} not {constraint}. Spec {spec_section}")

    elif constraint.startswith(">="):
        # Parse ">= a" format (must check before ">")
        threshold = float(constraint[2:].strip())
        if value < threshold:
            raise ValidationError(f"{name}={value} not {constraint}. Spec {spec_section}")

    elif constraint.startswith(">"):
        # Parse "> a" format
        threshold = float(constraint[1:].strip())
        if value <= threshold:
            raise ValidationError(f"{name}={value} not {constraint}. Spec {spec_section}")

    elif constraint.startswith("<="):
        # Parse "<= a" format (must check before "<")
        threshold = float(constraint[2:].strip())
        if value > threshold:
            raise ValidationError(f"{name}={value} not {constraint}. Spec {spec_section}")

    elif constraint.startswith("<"):
        # Parse "< a" format
        threshold = float(constraint[1:].strip())
        if value >= threshold:
            raise ValidationError(f"{name}={value} not {constraint}. Spec {spec_section}")


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


def format_constraint_summary() -> str:
    """Format human-readable summary of all constraints."""
    summary = get_constraint_summary()
    lines = [
        "",
        "=" * 70,
        "APGI PARAMETER CONSTRAINTS (Spec §15)",
        "=" * 70,
    ]

    for category, constraints in summary.items():
        lines.append(f"\n{category}:")
        for constraint in constraints:
            lines.append(f"  • {constraint}")

    lines.append("\n" + "=" * 70)
    return "\n".join(lines)


def print_constraint_summary() -> None:
    """Print human-readable summary of all constraints to stdout.

    Examples:
        >>> print_constraint_summary()  # Prints constraint summary
    """
    print(format_constraint_summary())
