from __future__ import annotations

import numpy as np


def compute_metabolic_cost(S: float, c0: float = 0.0, c1: float = 1.0) -> float:
    """C(t)=c0 + c1*S(t)."""

    return float(c0 + c1 * S)


def compute_metabolic_cost_realistic(
    S: float, B_prev: int, c1: float = 1.0, c2: float = 1.0
) -> float:
    """C(t)=c1*S(t)+c2*B(t-1)."""

    return float(c1 * S + c2 * B_prev)


def compute_information_value(
    z_e: float, z_i_eff: float, v1: float = 1.0, v2: float = 1.0
) -> float:
    """V(t)≈v1|z_e|+v2|z_i_eff|."""

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
