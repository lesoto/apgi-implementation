"""Asymmetric Signed Nonlinear Transformation φ(ε) — §6 of APGI Full Specs.

φ(ε) = α⁺ · tanh(γ⁺ · ε)   if ε ≥ 0   (reward / approach signal)
φ(ε) = α⁻ · tanh(γ⁻ · ε)   if ε < 0   (threat / avoidance signal)

Parameter bounds (§6, Parameter Bounds and Derivation):
  α⁺, α⁻  ∈ [0.5, 2.0]  — valence-specific amplitude gain
  γ⁺, γ⁻  ∈ [1.0, 5.0]  — saturation steepness

Symmetric first-order approximation (α⁺=α⁻, γ⁺=γ⁻) recovers the unsigned
|ε| formulation used in earlier APGI iterations (§6 Compatibility Note).
"""

from __future__ import annotations

import numpy as np

# Spec §6 parameter bounds
ALPHA_MIN: float = 0.5
ALPHA_MAX: float = 2.0
GAMMA_MIN: float = 1.0
GAMMA_MAX: float = 5.0


def validate_phi_params(
    alpha_pos: float,
    alpha_neg: float,
    gamma_pos: float,
    gamma_neg: float,
) -> None:
    """Raise ValueError if any parameter is outside spec-mandated bounds."""
    if not (ALPHA_MIN <= alpha_pos <= ALPHA_MAX):
        raise ValueError(f"alpha_plus={alpha_pos} outside [{ALPHA_MIN}, {ALPHA_MAX}]")
    if not (ALPHA_MIN <= alpha_neg <= ALPHA_MAX):
        raise ValueError(f"alpha_minus={alpha_neg} outside [{ALPHA_MIN}, {ALPHA_MAX}]")
    if not (GAMMA_MIN <= gamma_pos <= GAMMA_MAX):
        raise ValueError(f"gamma_plus={gamma_pos} outside [{GAMMA_MIN}, {GAMMA_MAX}]")
    if not (GAMMA_MIN <= gamma_neg <= GAMMA_MAX):
        raise ValueError(f"gamma_minus={gamma_neg} outside [{GAMMA_MIN}, {GAMMA_MAX}]")


def phi_transform(
    epsilon: float,
    alpha_pos: float = 1.0,
    alpha_neg: float = 1.0,
    gamma_pos: float = 2.0,
    gamma_neg: float = 2.0,
) -> float:
    """Signed asymmetric tanh transform of a scalar prediction error.

    Returns a signed float in (-alpha_pos, alpha_pos) for ε≥0
    and in (alpha_neg·(-1), 0) for ε<0.
    """
    if epsilon >= 0.0:
        return float(alpha_pos * np.tanh(gamma_pos * epsilon))
    return float(alpha_neg * np.tanh(gamma_neg * epsilon))


def phi_transform_array(
    epsilon_arr: np.ndarray,
    alpha_pos: float = 1.0,
    alpha_neg: float = 1.0,
    gamma_pos: float = 2.0,
    gamma_neg: float = 2.0,
) -> np.ndarray:
    """Vectorized signed asymmetric tanh transform over a NumPy array."""
    arr = np.asarray(epsilon_arr, dtype=float)
    pos_branch = alpha_pos * np.tanh(gamma_pos * arr)
    neg_branch = alpha_neg * np.tanh(gamma_neg * arr)
    return np.where(arr >= 0.0, pos_branch, neg_branch)
