from __future__ import annotations

from collections.abc import Callable

import numpy as np

#: Canonical precision clamp bounds (MathSpec §2: "Bounds: Π_min = 0.1,
#: Π_max = 10, matching the canonical precision range [0.1, 10] AU across all
#: APGI documents"). The previous default of Π_min = 0.01 was an order of
#: magnitude below the spec floor while the docstring asserted it *was* the
#: spec value, so a precision an order of magnitude outside the canonical
#: range could propagate into S_inst unflagged.
PI_MIN_CANONICAL: float = 0.1
PI_MAX_CANONICAL: float = 10.0


def clamp(value: float, lower: float, upper: float) -> float:
    if lower > upper:
        raise ValueError("lower must be <= upper")
    return float(max(lower, min(upper, value)))


def compute_precision(
    sigma2: float,
    eps: float = 1e-8,
    pi_min: float = PI_MIN_CANONICAL,
    pi_max: float = PI_MAX_CANONICAL,
) -> float:
    """Precision Π = 1/(σ²+ε), clamped to [Π_min, Π_max].

    Spec-mandated bounds: Π_min = 0.1, Π_max = 10 (MathSpec §2). "The clamp
    must be applied after every precision computation step."
    """
    raw = 1.0 / (max(float(sigma2), 0.0) + eps)
    return clamp(raw, pi_min, pi_max)


def update_mean_ema(prev_mean: float, z: float, alpha: float) -> float:
    """Online EMA mean: μ(t+1) = (1-α)μ(t) + α·z(t)."""
    if not (0.0 < alpha <= 1.0):
        raise ValueError("alpha must be in (0,1]")
    return float((1.0 - alpha) * prev_mean + alpha * z)


def update_uncertainty_phi(prev_sigma2: float, phi: float, alpha: float) -> float:
    """Online EMA uncertainty update (§7.1, Eq 158).
    Σ(t+1) = (1-α)Σ(t) + α · φ(ε)²
    Uses the squared asymmetric phi transform instead of raw squared deviation.
    This ensures that the uncertainty tracker (and thus precision) is
    valence-sensitive.
    Args:
        prev_sigma2: Previous uncertainty estimate Σ(t)
        phi: Current phi-transformed prediction error φ(ε)
        alpha: Learning rate (1/τ_Σ)
    Returns:
        Updated uncertainty estimate Σ(t+1)
    """
    if not (0.0 < alpha <= 1.0):
        raise ValueError("alpha must be in (0,1]")
    return float((1.0 - alpha) * prev_sigma2 + alpha * phi**2)


def update_variance_ema(prev_sigma2: float, z: float, mu: float, alpha: float) -> float:
    """[LEGACY] Online EMA variance: σ²(t+1) = (1-α)σ²(t) + α·(z-μ)².
    Maintained for backward compatibility. New code should use update_uncertainty_phi
    to satisfy Spec §7.1.
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


def apply_serotonin_threshold_patience(theta: float, beta_5ht: float) -> float:
    """Serotonergic patience/uncertainty-tolerance offset on θ: θ_eff = θ + β_5HT (§8.4).
    5-HT raises the ignition threshold, implementing 'patience' — the agent
    waits for stronger evidence before firing.  β_5HT = 0 is the default
    (no serotonergic modulation).  Positive values model SSRI-like states;
    negative values model 5-HT depletion or urgency.
    Unlike ACh (precision gain on Π_e) and NE (precision gain on Π_i), 5-HT
    acts on the threshold node rather than the precision node, but belongs in
    the neuromodulation layer because it controls the same evidence-gating
    decision at the ignition step.
    """
    return float(theta + beta_5ht)


def compute_interoceptive_precision_exponential(
    pi_baseline: float,
    beta_somatic: float,
    M: float,
    pi_min: float = PI_MIN_CANONICAL,
    pi_max: float = PI_MAX_CANONICAL,
) -> float:
    """Compute interoceptive precision with exponential somatic modulation.
    Formula: Π_i_eff = Π_i_baseline · exp(β_SM · M(c,a))
    This is the CANONICAL somatic-marker gain β_SM (Notation Appendix
    canonical range [0.1, 1.2], typical waking sub-band [0.3, 0.8]) — kept
    as the parameter name `beta_somatic` for backward compatibility, but
    distinct from the *linear-form* weight the Notation Appendix separately
    calls β_somatic (canonical range [0.5, 2.5], used in the alternate
    reduced form S += β_somatic·Π_i·|ε_i|, valid only when |β·M| << 1 —
    not implemented by this function). Also distinct from β_DA (dopaminergic
    additive error bias, unconstrained real — see apply_dopamine_bias_to_error).
    Where:
    - Π_i_baseline: Baseline precision from variance estimation
    - β_SM: Somatic-marker exponential gain, canonical [0.1, 1.2]
    - M: Somatic marker ∈ [-2, +2]
    The clamp is applied AFTER the exponential (not before) so it actually
    bounds the blow-up: at β_SM=2.5, M=+2, exp(5)≈148, which the clamp caps
    to pi_max.
    Args:
        pi_baseline: Baseline precision value
        beta_somatic: β_SM, the exponential somatic-marker gain
        M: Somatic marker value ∈ [-2, +2]
        pi_min: Minimum precision (clamping)
        pi_max: Maximum precision (clamping)
    Returns:
        Exponentially modulated precision, clamped to [pi_min, pi_max]
    """
    modulation = np.exp(beta_somatic * M)
    pi_eff = pi_baseline * modulation
    return float(np.clip(pi_eff, pi_min, pi_max))


def precision_coupling_ode_core(
    pi_ell: float,
    tau_pi: float,
    epsilon_ell: float,
    alpha_gain: float,
    pi_ell_plus_1: float | None,
    epsilon_ell_minus_1: float | None,
    C_down: float,
    C_up: float,
    psi: Callable[[float], float] | None = None,
) -> float:
    """Compute dΠ_ℓ/dt for hierarchical precision coupling.
    Formula:
    dΠ_ℓ/dt = -Π_ℓ/τ_Π + α|ϵ_ℓ| + C_down(Π_{ℓ+1} - Π_ℓ) + C_up·ψ(ϵ_{ℓ-1})
    Components:
    - -Π_ℓ/τ_Π: Self-decay of precision
    - α|ϵ_ℓ|: Error-driven precision gain
    - C_down(Π_{ℓ+1} - Π_ℓ): Top-down precision coupling
    - C_up·ψ(ϵ_{ℓ-1}): Bottom-up error coupling
    Args:
    - pi_ell: Current level precision
    - tau_pi: Precision decay time constant
    - epsilon_ell: Current level prediction error
    - alpha_gain: Error-to-precision gain
    - pi_ell_plus_1: Higher level precision (None for top level)
    - epsilon_ell_minus_1: Lower level prediction error (None for bottom level)
    - C_down: Top-down coupling strength
    - C_up: Bottom-up coupling strength
    - psi: Nonlinear error transfer function ψ
    Returns:
    - dΠ_ℓ/dt (precision change rate)
    """
    decay = -pi_ell / tau_pi
    error_drive = alpha_gain * abs(epsilon_ell)
    top_down = 0.0
    if pi_ell_plus_1 is not None:
        top_down = C_down * (pi_ell_plus_1 - pi_ell)
    bottom_up = 0.0
    if epsilon_ell_minus_1 is not None:
        # SIGNED, not rectified. MathSpec §2: "ψ(ε) = tanh(γ_ψ·ε) is a
        # monotonically increasing, SIGNED function of the lower-level
        # prediction error, with ψ: ℝ → (−1, +1) and ψ(0) = 0, so that
        # bottom-up cross-level drive is absent when prediction error is zero
        # and NEGATIVE WHEN THE ERROR IS NEGATIVE."
        #
        # Taking abs() before ψ destroyed exactly that property: it made
        # dΠ/dt identical for ε = +1 and ε = −1, collapsing the signed
        # bottom-up channel into a rectified one and silently removing the
        # spec's inhibitory pathway. Note that the α_Π·|ε_ℓ| drive term above
        # IS unsigned by design — the spec writes it with explicit bars — so
        # the two terms differ deliberately and must not be made uniform.
        error_lower = float(epsilon_ell_minus_1)
        if psi is not None:
            error_lower = psi(error_lower)
        bottom_up = C_up * error_lower
    return float(decay + error_drive + top_down + bottom_up)


def update_precision_euler(
    pi: float,
    dpi_dt: float,
    dt: float,
    pi_min: float = PI_MIN_CANONICAL,
    pi_max: float = PI_MAX_CANONICAL,
) -> float:
    """Update precision using Euler integration: Π(t+dt) = Π(t) + dt·dΠ/dt."""
    pi_new = pi + dt * dpi_dt
    return float(np.clip(pi_new, pi_min, pi_max))
