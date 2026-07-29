from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .rng import RNGLike, resolve_rng

NumberOrFn = float | Callable[[float, float], float]


def integrate_euler_maruyama(
    x: float, mu: NumberOrFn, sigma: NumberOrFn, t: float, dt: float, rng: RNGLike = None
) -> float:
    """Euler–Maruyama: X_{t+1} = X_t + μ(X_t,t)dt + σ(X_t,t)√dt N(0,1).

    The √dt scaling on the diffusion term is mandatory (MathSpec §7); omitting
    it produces incorrect noise scaling.

    Args:
        x: Current state X_t.
        mu: Drift μ(X, t) — scalar or callable.
        sigma: Diffusion σ(X, t) — scalar or callable.
        t: Current time.
        dt: Integration step, must be > 0.
        rng: Generator/seed for the Wiener increment; ``None`` uses the
            process-global APGI generator.

    Returns:
        X_{t+dt}.
    """
    if dt <= 0:
        raise ValueError("dt must be > 0")
    drift = float(mu(x, t) if callable(mu) else mu)
    diffusion = float(sigma(x, t) if callable(sigma) else sigma)
    noise = float(resolve_rng(rng).normal(0.0, 1.0))
    return float(x + drift * dt + diffusion * np.sqrt(dt) * noise)
