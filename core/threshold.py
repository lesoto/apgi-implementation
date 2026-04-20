from __future__ import annotations

import numpy as np


def compute_metabolic_cost(S: float, c0: float = 0.0, c1: float = 1.0) -> float:
    """C(t)=c0 + c1*S(t)."""

    return float(c0 + c1 * S)


def compute_metabolic_cost_realistic(S: float, B_prev: int, c1: float = 1.0, c2: float = 1.0) -> float:
    """C(t)=c1*S(t)+c2*B(t-1)."""

    return float(c1 * S + c2 * B_prev)


def compute_information_value(z_e: float, z_i: float, v1: float = 1.0, v2: float = 1.0) -> float:
    """V(t)≈v1|z_e|+v2|z_i|."""

    return float(v1 * abs(z_e) + v2 * abs(z_i))


def apply_ne_threshold_modulation(theta: float, g_ne: float, gamma_ne: float) -> float:
    """θ <- θ * (1 + γ_NE * g_NE)."""

    return float(theta * (1.0 + gamma_ne * g_ne))


def threshold_decay(theta: float, theta_base: float, kappa: float) -> float:
    """θ_next = θ_base + (θ-θ_base)e^{-κ}."""

    if kappa < 0:
        raise ValueError("kappa must be >= 0")
    return float(theta_base + (theta - theta_base) * np.exp(-kappa))


def update_threshold_discrete(theta: float, metabolic_cost: float, information_value: float, eta: float = 0.1) -> float:
    """Core: θ(t+1)=θ(t)+η[C(t)-V(t)]."""

    return float(theta + eta * (metabolic_cost - information_value))


def apply_refractory_boost(theta_next: float, B: int, delta: float) -> float:
    """Post-ignition boost: θ <- θ + δ*B."""

    return float(theta_next + delta * int(B))
