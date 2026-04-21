"""Somatic marker system for interoceptive precision modulation.

Implements M(c,a) somatic marker mapping physiological state to bounded
marker values ∈ [-2, +2], with exponential precision modulation.
"""

from __future__ import annotations

import numpy as np


def somatic_marker_arousal(arousal: float) -> float:
    """Map arousal level to somatic marker M(c,a) ∈ [-2, +2].

    Arousal is assumed to be normalized to [0, 1] where:
    - 0 = deep sleep/comatose (very low arousal)
    - 0.5 = calm/resting (baseline)
    - 1.0 = panic/fight-or-flight (very high arousal)

    Returns:
        M ∈ [-2, +2] where positive = elevated arousal, negative = suppressed
    """

    # Map [0, 1] to [-2, 2] with 0.5 → 0
    M = 4.0 * (arousal - 0.5)
    return float(np.clip(M, -2.0, 2.0))


def somatic_marker_valence(valence: float, arousal: float) -> float:
    """Combined valence-arousal mapping to somatic marker.

    Args:
        valence: Emotional valence ∈ [-1, 1] (negative to positive)
        arousal: Arousal level ∈ [0, 1] (low to high)

    Returns:
        M ∈ [-2, +2] combining both dimensions
    """

    # Circumplex model: M combines valence and arousal
    # High arousal amplifies valence effect
    M = 2.0 * valence * (0.5 + arousal)
    return float(np.clip(M, -2.0, 2.0))


def compute_precision_with_somatic_marker(
    pi_baseline: float,
    beta: float,
    M: float,
    pi_min: float = 1e-4,
    pi_max: float = 1e4,
) -> float:
    """Apply exponential somatic modulation to precision.

    Formula: Π^eff = Π^baseline · exp(β · M)

    Where:
    - Π^baseline: Baseline precision from variance estimation
    - β: Somatic bias parameter (typically 0.1-0.5)
    - M: Somatic marker ∈ [-2, +2]

    Args:
        pi_baseline: Baseline precision value
        beta: Somatic bias weight
        M: Somatic marker value ∈ [-2, +2]
        pi_min: Minimum precision (clamping)
        pi_max: Maximum precision (clamping)

    Returns:
        Exponentially modulated precision
    """

    # Exponential modulation
    modulation = np.exp(beta * M)
    pi_eff = pi_baseline * modulation

    # Clamp to stable range
    return float(np.clip(pi_eff, pi_min, pi_max))


def compute_somatic_gain(M: float, beta: float = 0.3) -> float:
    """Compute multiplicative gain from somatic marker.

    Returns exp(β · M) which can be used to modulate various parameters.

    Args:
        M: Somatic marker ∈ [-2, +2]
        beta: Modulation strength

    Returns:
        Gain factor ∈ [exp(-2β), exp(2β)]
    """

    return float(np.exp(beta * M))


def update_somatic_marker_euler(
    M_prev: float,
    arousal_target: float,
    tau_M: float = 500.0,
    dt: float = 1.0,
) -> float:
    """ODE update for somatic marker with arousal target.

    dM/dt = -(M - M_target)/τ_M

    Args:
        M_prev: Previous marker value
        arousal_target: Target arousal level [0, 1]
        tau_M: Time constant for marker dynamics (ms)
        dt: Time step (ms)

    Returns:
        Updated marker value
    """

    M_target = somatic_marker_arousal(arousal_target)
    dM = -(M_prev - M_target) / tau_M * dt
    M_new = M_prev + dM
    return float(np.clip(M_new, -2.0, 2.0))
