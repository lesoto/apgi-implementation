from __future__ import annotations

import numpy as np


def clamp(value: float, lower: float, upper: float) -> float:
    if lower > upper:
        raise ValueError("lower must be <= upper")
    return float(max(lower, min(upper, value)))


def compute_precision(
    sigma2: float,
    eps: float = 1e-8,
    pi_min: float = 1e-4,
    pi_max: float = 1e4,
) -> float:
    """Precision Π = 1/(σ²+ε), clamped to [Π_min, Π_max]."""

    raw = 1.0 / (max(float(sigma2), 0.0) + eps)
    return clamp(raw, pi_min, pi_max)


def update_mean_ema(prev_mean: float, z: float, alpha: float) -> float:
    """Online EMA mean: μ(t+1) = (1-α)μ(t) + α·z(t)."""

    if not (0.0 < alpha <= 1.0):
        raise ValueError("alpha must be in (0,1]")
    return float((1.0 - alpha) * prev_mean + alpha * z)


def update_variance_ema(prev_sigma2: float, z: float, mu: float, alpha: float) -> float:
    """Online EMA variance: σ²(t+1) = (1-α)σ²(t) + α·(z-μ)².

    Uses centered squared deviation to avoid overestimation when errors
    have nonzero mean.
    """

    if not (0.0 < alpha <= 1.0):
        raise ValueError("alpha must be in (0,1]")
    return float((1.0 - alpha) * prev_sigma2 + alpha * (z - mu) ** 2)


def apply_ach_gain(pi_e: float, g_ach: float) -> float:
    """Π_e_eff = g_ACh * Π_e."""

    return float(g_ach * pi_e)


def apply_ne_gain(pi_i: float, g_ne: float) -> float:
    """Π_i_eff = g_NE * Π_i (linear approximation)."""

    return float(g_ne * pi_i)


def apply_dopamine_bias_to_error(z_i: float, beta: float) -> float:
    """Dopamine additive bias on error: z_i_eff = z_i + β."""

    return float(z_i + beta)


def compute_effective_interoceptive_precision(
    pi_baseline: float,
    beta: float,
    somatic_marker: float,
) -> float:
    """Spec exponential form: Π_i_eff = Π_i_baseline · exp(β · M(c,a)).

    Models interoceptive precision as exponentially gated by the somatic
    marker M (e.g., interoceptive confidence or arousal state).
    """

    return float(pi_baseline * np.exp(beta * somatic_marker))


def update_precision_ode(
    pi: float,
    epsilon: float,
    pi_above: float,
    pi_below: float,
    tau_pi: float,
    alpha: float,
    c_down: float,
    c_up: float,
    psi_fn,
) -> float:
    """Hierarchical precision ODE (rate of change):
    dΠ/dt = -Π/τ_Π + α|ε| + c_down(Π_above - Π) + c_up·ψ(ε).

    Args:
        pi: Current precision at this level.
        epsilon: Local prediction error.
        pi_above: Precision from the level above (top-down).
        pi_below: Precision from the level below (unused in return, kept for
            symmetry with the full bidirectional form).
        tau_pi: Precision decay time constant.
        alpha: Error-driven precision update rate.
        c_down: Top-down coupling strength.
        c_up: Bottom-up nonlinear coupling strength.
        psi_fn: Nonlinear bottom-up gating function ψ(ε).
    """

    return float(
        -pi / tau_pi
        + alpha * abs(epsilon)
        + c_down * (pi_above - pi)
        + c_up * float(psi_fn(epsilon))
    )
