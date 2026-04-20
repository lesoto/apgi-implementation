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
