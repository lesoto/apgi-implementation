from __future__ import annotations

import numpy as np

from core.phi_transform import phi_transform


def instantaneous_signal(z_e: float, z_i_eff: float, pi_e_eff: float, pi_i_eff: float) -> float:
    """S_inst = Π_e_eff |z_e| + Π_i_eff |z_i_eff|.

    Symmetric |z| approximation — valid only as a baseline or non-affective
    context. For affective paradigms use instantaneous_signal_phi with
    pre-applied φ(ε) (§6).
    """

    return float(pi_e_eff * abs(z_e) + pi_i_eff * abs(z_i_eff))


def instantaneous_signal_with_dopamine(
    z_e: float,
    z_i: float,
    pi_e_eff: float,
    pi_i_eff: float,
    beta: float,
) -> float:
    """Alternative dopamine formula: S_inst = Π_e_eff |z_e| + Π_i_eff |z_i| + β.

    Symmetric |z| approximation — valid only as a baseline or non-affective
    context. Per spec Section 3.1: dopamine can act as bias on error
    (z_i_eff = z_i + β) OR as additive term in signal (S = Π_e|z_e| + Π_i|z_i| + β).
    For affective paradigms use compute_apgi_signal which applies φ(ε) (§6).
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


def instantaneous_signal_phi(
    phi_e: float,
    phi_i_eff: float,
    pi_e_eff: float,
    pi_i_eff: float,
) -> float:
    """S_inst = Π_e_eff · φ(z_e) + Π_i_eff · φ(z_i_eff).

    Signed version required by §12 (S_inst⁽ˡ⁾ = Π · φ(ε) · Γ).
    phi_e and phi_i_eff are already φ-transformed errors; no abs is taken.
    Negative values suppress ignition (avoidance pathway).
    """
    return float(pi_e_eff * phi_e + pi_i_eff * phi_i_eff)


def compute_apgi_signal(
    z_e: float,
    z_i: float,
    pi_e: float,
    pi_i_eff: float,
    beta: float = 0.0,
    dopamine_mode: str = "error_bias",
    alpha_pos: float = 1.0,
    alpha_neg: float = 1.0,
    gamma_pos: float = 2.0,
    gamma_neg: float = 2.0,
) -> float:
    """Convenience wrapper supporting both dopamine modes with φ(ε) transform (§6, §12).

    Applies the asymmetric valence-specific nonlinear transform φ(ε) to raw
    prediction errors before precision-weighting, replacing the symmetric |z|
    approximation mandated for affective paradigms.

    Args:
        z_e, z_i: Raw prediction errors
        pi_e, pi_i_eff: Precisions
        beta: Dopaminergic bias
        dopamine_mode: "error_bias" (z_i + β → φ) or "signal_additive" (φ(z_i) + β)
        alpha_pos, alpha_neg: Valence-specific amplitude gains α⁺, α⁻ ∈ [0.5, 2.0]
        gamma_pos, gamma_neg: Saturation steepness γ⁺, γ⁻ ∈ [1.0, 5.0]
    """

    if dopamine_mode == "error_bias":
        z_i_eff = z_i + beta
        phi_e = phi_transform(z_e, alpha_pos, alpha_neg, gamma_pos, gamma_neg)
        phi_i = phi_transform(z_i_eff, alpha_pos, alpha_neg, gamma_pos, gamma_neg)
        return instantaneous_signal_phi(phi_e, phi_i, pi_e, pi_i_eff)
    elif dopamine_mode == "signal_additive":
        phi_e = phi_transform(z_e, alpha_pos, alpha_neg, gamma_pos, gamma_neg)
        phi_i = phi_transform(z_i, alpha_pos, alpha_neg, gamma_pos, gamma_neg)
        return float(instantaneous_signal_phi(phi_e, phi_i, pi_e, pi_i_eff) + beta)
    else:
        raise ValueError(f"unknown dopamine_mode: {dopamine_mode}")
