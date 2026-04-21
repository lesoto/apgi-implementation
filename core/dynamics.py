from __future__ import annotations

import numpy as np


def update_signal_ode(
    S: float,
    z_e: float,
    z_i: float,
    pi_e: float,
    pi_i: float,
    beta: float,
    tau_s: float,
    noise_std: float = 0.01,
) -> float:
    """dS/dt = -S/τ_S + Π^e|z^e| + β Π^i |z^i| + η_S(t)."""

    if tau_s <= 0:
        raise ValueError("tau_s must be > 0")
    noise = float(np.random.normal(0.0, noise_std))
    return float(-S / tau_s + pi_e * abs(z_e) + beta * pi_i * abs(z_i) + noise)


def update_threshold_ode(
    theta: float,
    S: float,
    C: float,
    V: float,
    tau_theta: float,
    eta: float,
    gamma: float = 0.1,
    noise_std: float = 0.01,
) -> float:
    """Continuous threshold dynamics ODE.

    dθ/dt = -(θ - θ_base)/τ_θ + η·(C - V) + γ·S + η_θ(t)

    Components:
    - Decay term: -(θ - θ_base)/τ_θ (relaxation to baseline)
    - Allostatic term: η·(C - V) (cost-value mismatch)
    - Signal-driven term: γ·S (signal-dependent adaptation)
    - Noise term: η_θ(t) (stochastic fluctuations)

    Args:
        theta: Current threshold
        S: Current signal value
        C: Metabolic cost
        V: Information value
        tau_theta: Threshold decay time constant
        eta: Allostatic learning rate
        gamma: Signal-driven adaptation strength
        noise_std: Standard deviation of threshold noise

    Returns:
        dθ/dt (threshold change rate)
    """

    if tau_theta <= 0:
        raise ValueError("tau_theta must be > 0")

    # Decay toward baseline (implicit θ_base = 0 in ODE, handled externally)
    decay_term = -theta / tau_theta

    # Allostatic update (cost - value)
    allostatic_term = eta * (C - V)

    # Signal-driven adaptation
    signal_term = gamma * S

    # Stochastic noise
    noise = float(np.random.normal(0.0, noise_std))

    return float(decay_term + allostatic_term + signal_term + noise)
