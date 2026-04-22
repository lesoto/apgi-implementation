"""Hierarchical precision coupling and phase-locked threshold facilitation.

Implements:
1. Precision coupling ODE across hierarchy levels
2. Phase-locked threshold modulation via oscillatory coupling
3. Level count estimation from timescale range
"""

from __future__ import annotations

import numpy as np
from typing import Callable


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

    Formula: θ_{t,ℓ} = θ_{0,ℓ} · [1 + κ_down · Π_{ℓ+1} · cos(ϕ_{ℓ+1})]

    Higher levels modulate lower thresholds via oscillatory phase,
    creating rhythmic windows of opportunity for ignition.

    Args:
        theta_0_ell: Baseline threshold for level ℓ
        pi_ell_plus_1: Higher level precision
        phi_ell_plus_1: Phase of higher level oscillation (radians)
        kappa_down: Phase coupling strength
        phase_sensitivity: Additional phase sensitivity scaling

    Returns:
        Phase-modulated threshold
    """

    phase_modulation = 1.0 + kappa_down * pi_ell_plus_1 * np.cos(phi_ell_plus_1)

    # Apply phase sensitivity
    if phase_sensitivity != 1.0:
        # Nonlinear phase sensitivity
        phase_modulation = 1.0 + (phase_modulation - 1.0) * phase_sensitivity

    return float(theta_0_ell * phase_modulation)


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
                coupling_term += coupling_matrix[ell, j] * np.sin(
                    phi_array[j] - phi_array[ell]
                )
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
    ):
        """Initialize hierarchical precision network.

        Args:
            n_levels: Number of hierarchy levels
            taus: Timescales for each level (default: log-spaced)
            tau_pi: Precision decay time constant
            C_down: Top-down coupling strength
            C_up: Bottom-up coupling strength
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
    ) -> tuple[np.ndarray, np.ndarray]:
        """Update precision and phase dynamics across all levels.

        Args:
            epsilon_new: New prediction errors for each level
            dt: Time step
            alpha_gain: Error-to-precision gain

        Returns:
            Updated (precision, phase) arrays
        """

        self.epsilon = epsilon_new
        pi_new = np.zeros(self.n_levels)
        phi_new = np.zeros(self.n_levels)

        for ell in range(self.n_levels):
            # Precision coupling ODE
            pi_ell_plus_1 = self.pi[ell + 1] if ell < self.n_levels - 1 else None
            # Spec expects bottom-up coupling from lower level error ψ(ε_{ℓ-1})
            epsilon_ell_minus_1 = self.epsilon[ell - 1] if ell > 0 else None

            dpi_dt = precision_coupling_ode(
                pi_ell=self.pi[ell],
                tau_pi=self.tau_pi,
                epsilon_ell=self.epsilon[ell],
                alpha_gain=alpha_gain,
                pi_ell_plus_1=pi_ell_plus_1,
                epsilon_ell_minus_1=epsilon_ell_minus_1,
                C_down=self.C_down,
                C_up=self.C_up,
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

        # Ensure non-negative precision
        self.pi = np.maximum(pi_new, 0.01)
        self.phi = phi_new

        return self.pi.copy(), self.phi.copy()

    def compute_thresholds(
        self,
        theta_0: np.ndarray,
        S_levels: np.ndarray | None = None,
        kappa_down: float = 0.1,
        kappa_up: float = 0.0,
    ) -> np.ndarray:
        """Compute modulated thresholds for all levels including PAC and cascade.

        Args:
            theta_0: Baseline thresholds for each level
            S_levels: Current signal levels (for bottom-up cascade)
            kappa_down: Top-down phase coupling strength
            kappa_up: Bottom-up cascade strength

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

        return thetas
