"""Hierarchical precision coupling and phase-locked threshold facilitation.

Implements:
1. Precision coupling ODE across hierarchy levels
2. Phase-locked threshold modulation via oscillatory coupling
3. Level count estimation from timescale range
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def estimate_hierarchy_levels(
    tau_min: float,
    tau_max: float,
    k: float = 1.6,
) -> int:
    """Estimate optimal number of hierarchical levels L per APGI spec.

    Formula: L = floor( log(τ_max/τ_min) / log(k) ) + 1

    Args:
        tau_min: Minimum timescale (fastest level, e.g., 10ms)
        tau_max: Maximum timescale (slowest level, e.g., years)
        k: Timescale separation factor (default: 1.6)

    Returns:
        Exact number of hierarchy levels L
    """

    if tau_min <= 0 or tau_max <= 0 or k <= 1:
        raise ValueError("tau_min, tau_max must be > 0, k > 1")

    ratio = tau_max / tau_min
    L = int(np.floor(np.log(ratio) / np.log(k))) + 1

    return L


def precision_coupling_ode(
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
    """Compute dΠ_ℓ/dt for precision coupling across hierarchy.

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

    # Self-decay
    decay = -pi_ell / tau_pi

    # Error-driven gain
    error_drive = alpha_gain * abs(epsilon_ell)

    # Top-down coupling (from higher level)
    top_down = 0.0
    if pi_ell_plus_1 is not None:
        top_down = C_down * (pi_ell_plus_1 - pi_ell)

    # Bottom-up coupling (from lower level error)
    bottom_up = 0.0
    if epsilon_ell_minus_1 is not None:
        error_lower = abs(epsilon_ell_minus_1)
        if psi is not None:
            error_lower = psi(error_lower)
        bottom_up = C_up * error_lower

    return float(decay + error_drive + top_down + bottom_up)


def phase_locked_threshold(
    theta_0_ell: float,
    pi_ell_plus_1: float,
    phi_ell_plus_1: float,
    kappa_down: float,
    phase_sensitivity: float = 1.0,
) -> float:
    """Compute phase-locked threshold facilitation from higher level (§8.4).

    Formula: theta_l[t] = theta_0_l * (1 + kappa_down * precision_l_plus_1 * cos(phi_l_plus_1))

    Higher levels modulate lower thresholds via oscillatory phase,
    creating rhythmic windows of opportunity for ignition.

    Args:
        theta_0_ell: Baseline threshold for level ℓ
        pi_ell_plus_1: Higher level precision (precision_l_plus_1)
        phi_ell_plus_1: Phase of higher level oscillation (phi_l_plus_1)
        kappa_down: Phase coupling strength
        phase_sensitivity: Additional phase sensitivity scaling

    Returns:
        Phase-modulated threshold
    """

    # theta_l[t] = theta_0_l * (1 + kappa_down * precision_l_plus_1 * cos(phi_l_plus_1))
    phase_modulation = 1.0 + kappa_down * pi_ell_plus_1 * np.cos(phi_ell_plus_1)

    # Apply phase sensitivity
    if phase_sensitivity != 1.0:
        # Nonlinear phase sensitivity
        phase_modulation = 1.0 + (phase_modulation - 1.0) * phase_sensitivity

    return float(theta_0_ell * phase_modulation)


def nonlinear_phase_amplitude_coupling(
    theta_0_ell: float,
    pi_ell_plus_1: float,
    phi_ell_plus_1: float,
    kappa_down: float,
    nonlinearity: str = "sigmoid",
    phase_frequency_coupling: float = 0.0,
) -> float:
    """Compute nonlinear phase-amplitude coupling (PAC) with frequency modulation.

    Extends basic PAC with nonlinear frequency-amplitude coupling strength.

    Nonlinearity options:
    - 'sigmoid': Smooth saturation at high precision
    - 'power': Power-law frequency-amplitude coupling
    - 'exponential': Exponential frequency-amplitude coupling

    Formula (sigmoid):
    θ_mod = θ_{0,ℓ} · [1 + κ_down · σ(Π_{ℓ+1}) · cos(ϕ_{ℓ+1})]
    where σ(x) = 1 / (1 + exp(-x))

    Args:
        theta_0_ell: Baseline threshold for level ℓ
        pi_ell_plus_1: Higher level precision
        phi_ell_plus_1: Phase of higher level oscillation (radians)
        kappa_down: Phase coupling strength
        nonlinearity: Type of nonlinearity ('sigmoid', 'power', 'exponential')
        phase_frequency_coupling: Frequency-dependent coupling strength (0-1)

    Returns:
        Nonlinearly modulated threshold
    """

    # Apply nonlinearity to precision
    if nonlinearity == "sigmoid":
        # Sigmoid saturation: σ(x) = 1 / (1 + exp(-x))
        pi_nonlinear = 1.0 / (1.0 + np.exp(-pi_ell_plus_1))
    elif nonlinearity == "power":
        # Power-law: x^0.5 (sublinear)
        pi_nonlinear = np.sqrt(np.abs(pi_ell_plus_1))
    elif nonlinearity == "exponential":
        # Exponential: exp(x) - 1 (superlinear)
        pi_nonlinear = np.exp(np.clip(pi_ell_plus_1, -10, 10)) - 1.0
    else:
        pi_nonlinear = pi_ell_plus_1

    # Frequency-amplitude coupling: modulate coupling strength by phase
    # Higher frequency components get stronger coupling
    fac_strength = 1.0 + phase_frequency_coupling * np.abs(np.sin(phi_ell_plus_1))

    # Compute modulation
    phase_modulation = 1.0 + kappa_down * pi_nonlinear * np.cos(phi_ell_plus_1) * fac_strength

    return float(theta_0_ell * phase_modulation)


def bidirectional_phase_coupling(
    phi_ell: float,
    phi_ell_plus_1: float | None,
    phi_ell_minus_1: float | None,
    omega_ell: float,
    dt: float,
    kappa_down: float = 0.1,
    kappa_up: float = 0.05,
    noise_std: float = 0.0,
    rng: np.random.Generator | None = None,
) -> float:
    """Update phase with bidirectional Kuramoto coupling (top-down and bottom-up).

    Formula:
    dφ_ℓ/dt = ω_ℓ + κ_down·sin(φ_{ℓ+1} - φ_ℓ) + κ_up·sin(φ_{ℓ-1} - φ_ℓ) + ξ_ℓ(t)

    Args:
        phi_ell: Current phase at level ℓ
        phi_ell_plus_1: Phase at higher level (None for top level)
        phi_ell_minus_1: Phase at lower level (None for bottom level)
        omega_ell: Natural frequency at level ℓ
        dt: Time step
        kappa_down: Top-down coupling strength
        kappa_up: Bottom-up coupling strength
        noise_std: Phase noise standard deviation
        rng: Random number generator

    Returns:
        Updated phase (wrapped to [0, 2π])
    """

    generator = rng or np.random.default_rng()

    # Natural frequency term
    dphi = omega_ell * dt

    # Top-down coupling from higher level
    if phi_ell_plus_1 is not None and kappa_down > 0:
        dphi += kappa_down * np.sin(phi_ell_plus_1 - phi_ell) * dt

    # Bottom-up coupling from lower level
    if phi_ell_minus_1 is not None and kappa_up > 0:
        dphi += kappa_up * np.sin(phi_ell_minus_1 - phi_ell) * dt

    # Stochastic noise
    if noise_std > 0:
        dphi += noise_std * generator.normal(0.0, np.sqrt(dt))

    phi_new = (phi_ell + dphi) % (2 * np.pi)

    return float(phi_new)


def bottom_up_threshold_cascade(
    theta_ell: float,
    S_ell_minus_1: float,
    theta_ell_minus_1: float,
    kappa_up: float,
) -> float:
    """Compute bottom-up threshold cascade (§8.4).

    Formula: θ_ℓ ← θ_ℓ · [1 − κ_up · H(S_{ℓ−1} − θ_{ℓ−1})]
    where H is Heaviside function (1 if superthreshold, else 0).

    Args:
        theta_ell: Current threshold at level ℓ
        S_ell_minus_1: Signal at level ℓ-1
        theta_ell_minus_1: Threshold at level ℓ-1
        kappa_up: Cascade strength

    Returns:
        Modulated threshold
    """

    is_superthreshold = float(S_ell_minus_1 > theta_ell_minus_1)
    modulation = 1.0 - kappa_up * is_superthreshold

    return float(theta_ell * modulation)


def bidirectional_threshold_cascade(
    theta_ell: float,
    S_ell_minus_1: float | None,
    theta_ell_minus_1: float | None,
    S_ell_plus_1: float | None,
    theta_ell_plus_1: float | None,
    kappa_up: float = 0.1,
    kappa_down: float = 0.05,
    hysteresis: float = 0.1,
) -> float:
    """Compute bidirectional threshold cascade with hysteresis.

    Implements both bottom-up and top-down threshold modulation with
    hysteresis to prevent oscillations.

    Formula:
    θ_ℓ ← θ_ℓ · [1 − κ_up·H(S_{ℓ−1} − θ_{ℓ−1})] · [1 + κ_down·H(S_{ℓ+1} − θ_{ℓ+1})]

    Args:
        theta_ell: Current threshold at level ℓ
        S_ell_minus_1: Signal at level ℓ-1 (None for bottom level)
        theta_ell_minus_1: Threshold at level ℓ-1 (None for bottom level)
        S_ell_plus_1: Signal at level ℓ+1 (None for top level)
        theta_ell_plus_1: Threshold at level ℓ+1 (None for top level)
        kappa_up: Bottom-up cascade strength
        kappa_down: Top-down cascade strength
        hysteresis: Hysteresis factor to prevent oscillations

    Returns:
        Bidirectionally modulated threshold
    """

    modulation = 1.0

    # Bottom-up cascade: lower level ignition suppresses this level
    if S_ell_minus_1 is not None and theta_ell_minus_1 is not None:
        is_lower_superthreshold = float(S_ell_minus_1 > theta_ell_minus_1 * (1 - hysteresis))
        modulation *= 1.0 - kappa_up * is_lower_superthreshold

    # Top-down cascade: higher level ignition facilitates this level
    if S_ell_plus_1 is not None and theta_ell_plus_1 is not None:
        is_upper_superthreshold = float(S_ell_plus_1 > theta_ell_plus_1 * (1 + hysteresis))
        modulation *= 1.0 + kappa_down * is_upper_superthreshold

    return float(theta_ell * modulation)


def update_phase_dynamics(
    phi: float,
    omega: float,
    dt: float,
    coupling_strength: float = 0.0,
    phi_neighbor: float | None = None,
    noise_std: float = 0.0,
    rng: np.random.Generator | None = None,
) -> float:
    """Update oscillatory phase with Kuramoto coupling and noise (spec §9).

    Full Kuramoto model: dφ_ℓ/dt = ω_ℓ + Σ_j K_{ℓj} sin(φ_j - φ_ℓ) + ξ_ℓ(t)

    where ξ_ℓ(t) ~ N(0, σ²_ξ) is phase noise.

    Args:
        phi: Current phase (radians)
        omega: Natural frequency (rad/ms)
        dt: Time step (ms)
        coupling_strength: Phase coupling to neighbor (K)
        phi_neighbor: Neighboring phase for coupling (φ_j)
        noise_std: Standard deviation of phase noise (σ_ξ)
        rng: Random number generator for reproducibility

    Returns:
        Updated phase (wrapped to [0, 2π])
    """

    generator = rng or np.random.default_rng()

    # Natural frequency term
    dphi = omega * dt

    # Kuramoto-style phase coupling: K * sin(φ_j - φ_ℓ)
    if phi_neighbor is not None and coupling_strength != 0.0:
        dphi += coupling_strength * np.sin(phi_neighbor - phi) * dt

    # Stochastic noise term: ξ_ℓ(t) * sqrt(dt)
    if noise_std > 0:
        dphi += noise_std * generator.normal(0.0, np.sqrt(dt))

    phi_new = (phi + dphi) % (2 * np.pi)

    return float(phi_new)


def update_phase_kuramoto_full(
    phi_array: np.ndarray,
    omega_array: np.ndarray,
    coupling_matrix: np.ndarray,
    dt: float,
    noise_std: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Full Kuramoto coupling across all phases (spec §9).

    dφ_ℓ/dt = ω_ℓ + Σ_j K_{ℓj} sin(φ_j - φ_ℓ) + ξ_ℓ(t)

    Args:
        phi_array: Array of phases (L,) in radians
        omega_array: Array of natural frequencies (L,) in rad/ms
        coupling_matrix: Coupling strength matrix K[L×L] where K[i,j] couples j→i
        dt: Time step (ms)
        noise_std: Standard deviation of phase noise
        rng: Random number generator

    Returns:
        Updated phase array (L,) wrapped to [0, 2π]
    """

    generator = rng or np.random.default_rng()
    L = len(phi_array)
    phi_new = np.zeros_like(phi_array)

    for ell in range(L):
        # Natural frequency
        dphi = omega_array[ell] * dt

        # Sum over all coupled neighbors: Σ_j K_{ℓj} sin(φ_j - φ_ℓ)
        coupling_term = 0.0
        for j in range(L):
            if coupling_matrix[ell, j] != 0.0 and ell != j:
                coupling_term += coupling_matrix[ell, j] * np.sin(phi_array[j] - phi_array[ell])
        dphi += coupling_term * dt

        # Stochastic noise
        if noise_std > 0:
            dphi += noise_std * generator.normal(0.0, np.sqrt(dt))

        phi_new[ell] = (phi_array[ell] + dphi) % (2 * np.pi)

    return phi_new


class HierarchicalPrecisionNetwork:
    """Multi-level precision network with coupling dynamics."""

    def __init__(
        self,
        n_levels: int,
        taus: np.ndarray | None = None,
        tau_pi: float = 1000.0,
        C_down: float = 0.1,
        C_up: float = 0.05,
        psi: Callable[[float], float] | None = None,
    ):
        """Initialize hierarchical precision network.

        Args:
            n_levels: Number of hierarchy levels
            taus: Timescales for each level (default: log-spaced)
            tau_pi: Precision decay time constant
            C_down: Top-down coupling strength
            C_up: Bottom-up coupling strength
            psi: Nonlinear bottom-up transfer ψ for ε_{ℓ-1} (None → identity on |ε|)
        """

        self.n_levels = n_levels

        if taus is None:
            # Default: log-spaced from 10ms to ~10s
            self.taus = np.logspace(1, 4, n_levels)  # 10ms to 10s
        else:
            self.taus = np.array(taus)

        self.tau_pi = tau_pi
        self.C_down = C_down
        self.C_up = C_up
        self.psi = psi

        # Initialize state
        self.pi = np.ones(n_levels)  # Precision at each level
        self.phi = np.zeros(n_levels)  # Phase at each level
        self.omega = 2 * np.pi / self.taus  # Natural frequencies
        self.epsilon = np.zeros(n_levels)  # Prediction errors

    def step(
        self,
        epsilon_new: np.ndarray,
        dt: float = 1.0,
        alpha_gain: float = 0.1,
        epsilon_bottom_up: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Update precision and phase dynamics across all levels.

        Args:
            epsilon_new: New prediction errors for each level (drives α|ε_ℓ|)
            dt: Time step
            alpha_gain: Error-to-precision gain
            epsilon_bottom_up: Optional separate error series for ψ(ε_{ℓ-1}) coupling.
                If None, uses epsilon_new shifted by one level.

        Returns:
            Updated (precision, phase) arrays
        """

        self.epsilon = epsilon_new
        epsilon_bu = epsilon_bottom_up if epsilon_bottom_up is not None else self.epsilon
        pi_new = np.zeros(self.n_levels)
        phi_new = np.zeros(self.n_levels)

        for ell in range(self.n_levels):
            # Precision coupling ODE
            pi_ell_plus_1 = self.pi[ell + 1] if ell < self.n_levels - 1 else None
            # Spec expects bottom-up coupling from lower level error ψ(ε_{ℓ-1})
            epsilon_ell_minus_1 = epsilon_bu[ell - 1] if ell > 0 else None

            dpi_dt = precision_coupling_ode(
                pi_ell=self.pi[ell],
                tau_pi=self.tau_pi,
                epsilon_ell=self.epsilon[ell],
                alpha_gain=alpha_gain,
                pi_ell_plus_1=pi_ell_plus_1,
                epsilon_ell_minus_1=epsilon_ell_minus_1,
                C_down=self.C_down,
                C_up=self.C_up,
                psi=self.psi,
            )

            pi_new[ell] = self.pi[ell] + dt * dpi_dt

            # Phase update
            phi_neighbor = None
            if ell < self.n_levels - 1:
                phi_neighbor = self.phi[ell + 1]

            phi_new[ell] = update_phase_dynamics(
                phi=self.phi[ell],
                omega=self.omega[ell],
                dt=dt,
                coupling_strength=self.C_down * 0.1,  # Weak phase coupling
                phi_neighbor=phi_neighbor,
            )

        # Ensure non-negative precision and clamp to reasonable range (§7.4)
        # pi > 2.0 can cause threshold runaway in phase-locked modulation
        self.pi = np.clip(pi_new, 0.01, 2.0)
        self.phi = phi_new

        return self.pi.copy(), self.phi.copy()

    def compute_thresholds(
        self,
        theta_0: np.ndarray,
        S_levels: np.ndarray | None = None,
        kappa_down: float = 0.1,
        kappa_up: float = 0.0,
        theta_min: float = 0.1,
        theta_max: float = 20.0,
    ) -> np.ndarray:
        """Compute modulated thresholds for all levels including PAC and cascade.

        Args:
            theta_0: Baseline thresholds for each level
            S_levels: Current signal levels (for bottom-up cascade)
            kappa_down: Top-down phase coupling strength
            kappa_up: Bottom-up cascade strength
            theta_min: Minimum allowed threshold
            theta_max: Maximum allowed threshold

        Returns:
            Phase-modulated and cascade-modulated thresholds
        """

        thetas = np.zeros(self.n_levels)

        for ell in range(self.n_levels):
            # 1) Top-down Phase-Amplitude Coupling (§8.4)
            if ell < self.n_levels - 1:
                # Higher level modulates this level
                thetas[ell] = phase_locked_threshold(
                    theta_0_ell=theta_0[ell],
                    pi_ell_plus_1=self.pi[ell + 1],
                    phi_ell_plus_1=self.phi[ell + 1],
                    kappa_down=kappa_down,
                )
            else:
                # Top level: no top-down PAC
                thetas[ell] = theta_0[ell]

            # 2) Bottom-up Threshold Cascade (§8.4)
            if ell > 0 and S_levels is not None and kappa_up > 0:
                thetas[ell] = bottom_up_threshold_cascade(
                    theta_ell=thetas[ell],
                    S_ell_minus_1=S_levels[ell - 1],
                    theta_ell_minus_1=thetas[ell - 1],
                    kappa_up=kappa_up,
                )

        return np.clip(thetas, theta_min, theta_max)
