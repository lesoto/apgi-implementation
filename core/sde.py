from __future__ import annotations

from collections.abc import Callable

import numpy as np

NumberOrFn = float | Callable[[float, float], float]


def integrate_euler_maruyama(
    x: float, mu: NumberOrFn, sigma: NumberOrFn, t: float, dt: float
) -> float:
    """Euler–Maruyama: X_{t+1} = X_t + μ(X_t,t)dt + σ(X_t,t)√dt N(0,1)."""

    if dt <= 0:
        raise ValueError("dt must be > 0")

    drift = float(mu(x, t) if callable(mu) else mu)
    diffusion = float(sigma(x, t) if callable(sigma) else sigma)
    noise = float(np.random.normal(0.0, 1.0))
    return float(x + drift * dt + diffusion * np.sqrt(dt) * noise)
