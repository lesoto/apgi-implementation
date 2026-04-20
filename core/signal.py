from __future__ import annotations

import numpy as np


def instantaneous_signal(
    z_e: float, z_i_eff: float, pi_e_eff: float, pi_i_eff: float
) -> float:
    """S_inst = Π_e_eff |z_e| + Π_i_eff |z_i_eff|."""

    return float(pi_e_eff * abs(z_e) + pi_i_eff * abs(z_i_eff))


def instantaneous_signal_with_dopamine(
    z_e: float,
    z_i: float,
    pi_e_eff: float,
    pi_i_eff: float,
    beta: float,
) -> float:
    """Alternative dopamine formula: S_inst = Π_e_eff |z_e| + Π_i_eff |z_i| + β.

    Per spec Section 3.1: dopamine can act as bias on error (z_i_eff = z_i + β)
    OR as additive term in signal (S = Π_e|z_e| + Π_i|z_i| + β).
    """

    return float(pi_e_eff * abs(z_e) + pi_i_eff * abs(z_i) + beta)


def integrate_signal_leaky(S_prev: float, S_inst: float, lam: float) -> float:
    """S(t+1) = (1-λ)S(t) + λS_inst(t), λ in (0,1)."""

    if not (0.0 < lam < 1.0):
        raise ValueError("lam must be in (0,1)")
    return float((1.0 - lam) * S_prev + lam * S_inst)


def stabilize_signal_log(S: float, enabled: bool = True) -> float:
    """Optional stabilization: S <- log(1+S)."""

    if not enabled:
        return float(S)
    return float(np.log1p(max(0.0, S)))


def compute_apgi_signal(
    z_e: float,
    z_i: float,
    pi_e: float,
    pi_i_eff: float,
    beta: float = 0.0,
    dopamine_mode: str = "error_bias",
) -> float:
    """Convenience wrapper supporting both dopamine modes.

    Args:
        z_e, z_i: Prediction errors
        pi_e, pi_i_eff: Precisions
        beta: Dopaminergic bias
        dopamine_mode: "error_bias" (z_i + β) or "signal_additive" (S + β)
    """

    if dopamine_mode == "error_bias":
        z_i_eff = z_i + beta
        return instantaneous_signal(z_e, z_i_eff, pi_e, pi_i_eff)
    elif dopamine_mode == "signal_additive":
        return instantaneous_signal_with_dopamine(z_e, z_i, pi_e, pi_i_eff, beta)
    else:
        raise ValueError(f"unknown dopamine_mode: {dopamine_mode}")
