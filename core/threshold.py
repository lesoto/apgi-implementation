from __future__ import annotations

import numpy as np
from .thermodynamics import compute_landauer_cost, K_BOLTZMANN, T_ENV_DEFAULT


def compute_metabolic_cost(S: float, c0: float = 0.0, c1: float = 1.0) -> float:
    """C(t)=c0 + c1*S(t)."""

    return float(c0 + c1 * S)


def compute_metabolic_cost_realistic(
    S: float,
    B_prev: int,
    c1: float = 1.0,
    c2: float = 1.0,
    eps_stab: float = 1e-6,
    enforce_landauer: bool = False,
    kappa_meta: float = 1.0,
) -> float:
    """C(t)=c1*S(t)+c2*B(t-1) with optional Landauer constraint (spec §11).

    When enforce_landauer=True, ensures C(t) ≥ E_min per Landauer's principle:
        C(t) = max(c1·S + c2·B_prev, κ_meta·N_erase·k_B·T_env·ln(2))

    Args:
        S: Signal magnitude
        B_prev: Previous ignition state
        c1: Signal cost coefficient
        c2: Ignition cost coefficient
        eps_stab: Stability threshold for bit estimation
        enforce_landauer: Whether to enforce thermodynamic constraint
        kappa_meta: Metabolic efficiency factor (default: 1.0)

    Returns:
        Metabolic cost C(t)
    """

    base_cost = c1 * S + c2 * B_prev

    if enforce_landauer and S > eps_stab:
        # Compute Landauer minimum
        e_min = compute_landauer_cost(
            S=S,
            eps=eps_stab,
            k_b=K_BOLTZMANN,
            T_env=T_ENV_DEFAULT,
            kappa_meta=kappa_meta,
        )
        # Ensure cost meets thermodynamic minimum
        # Scale factor to convert Joules to dimensionless cost units
        # Using a scaling factor of 1e20 for neural-scale computations
        scale_factor = 1e20
        e_min_scaled = e_min * scale_factor
        base_cost = max(base_cost, e_min_scaled)

    return float(base_cost)


def compute_information_value(
    z_e: float, z_i_eff: float, v1: float = 1.0, v2: float = 1.0
) -> float:
    """V(t)≈v1|z_e|+v2|z_i_eff| (spec §4.3).

    Note: z_i_eff should include dopaminergic bias: z_i_eff = z_i + β_DA.
    This coupling makes ∂θ/∂β_DA = -η·v₂ < 0, implementing the established
    DA/motivation relationship (Berridge & Kringelbach, 2015).

    Args:
        z_e: Exteroceptive z-scored error
        z_i_eff: Interoceptive z-scored error WITH dopamine bias (z_i + β_DA)
        v1: Value coefficient for exteroceptive errors
        v2: Value coefficient for interoceptive errors

    Returns:
        Information value V(t)
    """

    return float(v1 * abs(z_e) + v2 * abs(z_i_eff))


def compute_information_value_with_bias(
    z_e: float, z_i: float, beta_da: float, v1: float = 1.0, v2: float = 1.0
) -> float:
    """V(t) with explicit dopamine bias (spec §4.3).

    V(t) = v₁·|z_e| + v₂·|z_i + β_DA|

    This form makes the dopaminergic coupling explicit.

    Args:
        z_e: Exteroceptive z-scored error
        z_i: Interoceptive z-scored error (without bias)
        beta_da: Dopaminergic additive bias
        v1: Value coefficient for exteroceptive errors
        v2: Value coefficient for interoceptive errors

    Returns:
        Information value V(t)
    """

    z_i_eff = z_i + beta_da
    return float(v1 * abs(z_e) + v2 * abs(z_i_eff))


def apply_ne_threshold_modulation(theta: float, g_ne: float, gamma_ne: float) -> float:
    """θ <- θ * (1 + γ_NE * g_NE)."""

    return float(theta * (1.0 + gamma_ne * g_ne))


def threshold_decay(theta: float, theta_base: float, kappa: float) -> float:
    """θ_next = θ_base + (θ-θ_base)e^{-κ}."""

    if kappa < 0:
        raise ValueError("kappa must be >= 0")
    return float(theta_base + (theta - theta_base) * np.exp(-kappa))


def update_threshold_discrete(
    theta: float,
    metabolic_cost: float,
    information_value: float,
    eta: float = 0.1,
    delta: float = 0.5,
    B_prev: int = 0,
) -> float:
    """Core allostatic update per APGI spec Section 4.

    Formula: θ(t+1) = θ(t) + η[C(t) - V(t)] + δ_reset·B(t)

    The refractory boost δ_reset·B is part of the core allostatic update,
    applied BEFORE NE modulation and ignition decision.

    Args:
        theta: Current threshold
        metabolic_cost: C(t) - metabolic cost of signal/ignition
        information_value: V(t) - expected information value of errors
        eta: Allostatic learning rate
        delta: Refractory boost magnitude (δ_reset)
        B_prev: Previous ignition state (0 or 1)

    Returns:
        Updated threshold
    """

    return float(theta + eta * (metabolic_cost - information_value) + delta * B_prev)


def apply_refractory_boost(theta_next: float, B: int, delta: float) -> float:
    """Post-ignition boost: θ <- θ + δ*B."""

    return float(theta_next + delta * int(B))


def update_threshold_ode_deprecated(
    theta: float,
    theta_0: float,
    dS_dt: float,
    B_prev: int,
    gamma: float,
    delta: float,
    lam: float,
) -> float:
    """[DEPRECATED] Continuous threshold dynamics (rate of change):
    dθ/dt = γ(θ_0 - θ) + δ·B(t-1) - λ|dS/dt|.

    Warning: Derivative coupling to dS/dt is removed from the APGI spec.
    Use core.allostatic.allostatic_threshold_ode instead.
    """

    return float(gamma * (theta_0 - theta) + delta * int(B_prev) - lam * abs(dS_dt))
