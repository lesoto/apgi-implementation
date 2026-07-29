from __future__ import annotations

import numpy as np

from .rng import RNGLike, resolve_rng


def compute_ignition_probability(S: float, theta: float, tau: float = 1.0) -> float:
    """P_ignite = sigmoid((S-θ)/τ)."""
    if tau <= 0:
        raise ValueError("tau must be > 0")
    x = (S - theta) / tau
    # Clamp to prevent overflow in exp
    x = np.clip(x, -500, 500)
    return float(1.0 / (1.0 + np.exp(-x)))


def sample_ignition_state(p_ignite: float, rng: RNGLike = None) -> int:
    """B ~ Bernoulli(P_ignite).

    Args:
        p_ignite: Ignition probability P_ign(t) ∈ [0, 1].
        rng: Generator, int seed, or ``None`` to use the process-global APGI
            generator (see :mod:`core.rng`). Never constructs an unseeded
            generator — that would make ignition sampling irreproducible.

    Returns:
        Bernoulli sample B(t) ∈ {0, 1}.
    """
    if not (0.0 <= p_ignite <= 1.0):
        raise ValueError("p_ignite must be in [0,1]")
    generator = resolve_rng(rng)
    return int(generator.random() < p_ignite)


def detect_ignition_event(S: float, theta: float) -> bool:
    return bool(S > theta)


def compute_margin(S: float, theta: float) -> float:
    return float(S - theta)
