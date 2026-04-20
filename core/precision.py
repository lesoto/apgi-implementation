from __future__ import annotations


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


def update_variance_ema(prev_sigma2: float, z: float, alpha: float) -> float:
    """Online EMA variance: σ²(t+1)=(1-α)σ²(t)+α z²(t)."""

    if not (0.0 < alpha <= 1.0):
        raise ValueError("alpha must be in (0,1]")
    return float((1.0 - alpha) * prev_sigma2 + alpha * (z**2))


def apply_ach_gain(pi_e: float, g_ach: float) -> float:
    """Π_e_eff = g_ACh * Π_e."""

    return float(g_ach * pi_e)


def apply_ne_gain(pi_i: float, g_ne: float) -> float:
    """Π_i_eff = g_NE * Π_i."""

    return float(g_ne * pi_i)


def apply_dopamine_bias_to_error(z_i: float, beta: float) -> float:
    """Dopamine correction: z_i_eff = z_i + β (not precision scaling)."""

    return float(z_i + beta)
