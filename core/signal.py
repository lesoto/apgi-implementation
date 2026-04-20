from __future__ import annotations

import numpy as np


def instantaneous_signal(z_e: float, z_i_eff: float, pi_e_eff: float, pi_i_eff: float) -> float:
    """S_inst = Π_e_eff |z_e| + Π_i_eff |z_i_eff|."""

    return float(pi_e_eff * abs(z_e) + pi_i_eff * abs(z_i_eff))


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


def compute_apgi_signal(z_e: float, z_i: float, pi_e: float, pi_i_eff: float) -> float:
    """Backward-compatible convenience wrapper."""

    return instantaneous_signal(z_e=z_e, z_i_eff=z_i, pi_e_eff=pi_e, pi_i_eff=pi_i_eff)
