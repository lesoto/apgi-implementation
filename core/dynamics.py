from __future__ import annotations

import numpy as np


def signal_drift(
    S: float,
    z_e: float,
    z_i: float,
    pi_e: float,
    pi_i: float,
    beta: float,
    tau_s: float,
) -> float:
    """Deterministic ODE drift (no noise): dS/dt|_det = -S/τ_S + Π_e|z_e| + Π_i|z_i_eff|.

    Separates the deterministic component so it can be passed as the drift
    argument to integrate_euler_maruyama in sde.py.
    """

    if tau_s <= 0:
        raise ValueError("tau_s must be > 0")

    # z_i_eff includes the dopamine bias: z_i_eff = z_i + beta
    z_i_eff = z_i + beta

    return float(-S / tau_s + pi_e * abs(z_e) + pi_i * abs(z_i_eff))


def update_signal_ode(
    S: float,
    z_e: float,
    z_i: float,
    pi_e: float,
    pi_i: float,
    beta: float,
    tau_s: float,
    dt: float = 1.0,
    noise_std: float = 0.01,
) -> float:
    """dS/dt = -S/τ_S + Π^e|z^e| + Π^i|z^i_eff| + η_S(t).

    Implements Euler-Maruyama integration step:
    S(t+dt) = S(t) + dS/dt * dt + noise_std * sqrt(dt) * N(0,1)
    """

    if tau_s <= 0:
        raise ValueError("tau_s must be > 0")

    # z_i_eff includes the dopamine bias: z_i_eff = z_i + beta
    z_i_eff = z_i + beta

    drift = -S / tau_s + pi_e * abs(z_e) + pi_i * abs(z_i_eff)
    noise = float(np.random.normal(0.0, noise_std * np.sqrt(dt)))

    return float(S + drift * dt + noise)


def compute_precision_coupled_noise_std(pi_e_eff: float, pi_i_eff: float) -> float:
    """σ_S = 1 / sqrt(Π_e^eff + Π_i^eff) (§7.3)."""

    total_pi = pi_e_eff + pi_i_eff
    if total_pi <= 0:
        return 1.0  # Default to high noise if precision is zero
    return float(1.0 / np.sqrt(total_pi))


def update_prediction(
    x_hat: float,
    epsilon: float,
    pi: float,
    kappa: float,
    pi_max: float = 100.0,
) -> float:
    """Generative Model Dynamics (Prediction Update) per §1.4.

    x̂(t+1) = x̂(t) + κ · Π(t) · ε(t)

    Stability condition: κ < 2 / Π_max ensures gradient descent convergence.

    Args:
        x_hat: Current prediction
        epsilon: Raw prediction error (x - x_hat)
        pi: Precision (Bayesian confidence)
        kappa: Prediction learning rate
        pi_max: Maximum precision clamp (for stability check)

    Returns:
        Updated prediction x_hat(t+1)
    """

    # Optional stability check: κ < 2 / Π_max
    # We don't enforce it strictly here but document it.
    return float(x_hat + kappa * pi * epsilon)


def update_threshold_ode(
    theta: float,
    theta_base: float,
    C: float,
    V: float,
    tau_theta: float,
    eta: float,
    dt: float = 1.0,
    noise_std: float = 0.01,
) -> float:
    """Continuous threshold dynamics ODE per APGI spec (§7.2).

    dθ/dt = -(θ - θ_base)/τ_θ + η·(C - V) + η_θ(t)

    Implements Euler-Maruyama integration step:
    θ(t+dt) = θ(t) + dθ/dt * dt + noise_std * sqrt(dt) * N(0,1)

    Components:
    - Decay term: -(θ - θ_base)/τ_θ (mean-reversion to baseline)
    - Allostatic term: η·(C - V) (cost-value mismatch)
    - Noise term: η_θ(t) (stochastic fluctuations)

    Args:
        theta: Current threshold
        theta_base: Baseline threshold value
        C: Metabolic cost
        V: Information value
        tau_theta: Threshold decay time constant (τ_θ)
        eta: Allostatic learning rate
        noise_std: Standard deviation of threshold noise

    Returns:
        dθ/dt (threshold change rate)
    """

    if tau_theta <= 0:
        raise ValueError("tau_theta must be > 0")

    # Mean-reversion toward baseline: -(θ - θ_base)/τ_θ
    decay_term = -(theta - theta_base) / tau_theta

    # Allostatic update (cost - value)
    allostatic_term = eta * (C - V)

    drift = decay_term + allostatic_term
    noise = float(np.random.normal(0.0, noise_std * np.sqrt(dt)))

    return float(theta + drift * dt + noise)
