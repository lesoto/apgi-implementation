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
    overlap_factor: float = 2.0,
) -> int:
    """Estimate optimal number of hierarchical levels.

    Formula: N ≈ log(τ_max/τ_min) / log(overlap_factor)

    Args:
        tau_min: Minimum timescale (fastest level, e.g., 10ms)
        tau_max: Maximum timescale (slowest level, e.g., years)
        overlap_factor: Overlap between adjacent levels (typically 2-3)

    Returns:
        Recommended number of hierarchy levels
    """

    if tau_min <= 0 or tau_max <= 0 or overlap_factor <= 1:
        raise ValueError("tau_min, tau_max must be > 0, overlap_factor > 1")

    ratio = tau_max / tau_min
    N = np.log10(ratio) / np.log10(overlap_factor)

    return int(np.ceil(N))


def precision_coupling_ode(
    pi_ell: float,
    tau_pi: float,
    epsilon_ell: float,
    alpha_gain: float,
    pi_ell_plus_1: float | None,
    pi_ell_minus_1: float | None,
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
        pi_ell: Current level precision
        tau_pi: Precision decay time constant
        epsilon_ell: Current level prediction error
        alpha_gain: Error-to-precision gain
        pi_ell_plus_1: Higher level precision (None for top level)
        pi_ell_minus_1: Lower level precision (None for bottom level)
        C_down: Top-down coupling strength
        C_up: Bottom-up coupling strength
        psi: Nonlinear error transfer function

    Returns:
        dΠ_ℓ/dt (precision change rate)
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
    if pi_ell_minus_1 is not None:
        error_lower = 0.0  # Simplified: use error from lower level
        if psi is not None:
            error_lower = psi(abs(epsilon_ell))  # Use current error as proxy
        else:
            error_lower = abs(epsilon_ell)
        bottom_up = C_up * error_lower

    return float(decay + error_drive + top_down + bottom_up)


def phase_locked_threshold(
    theta_0_ell: float,
    pi_ell_plus_1: float,
    phi_ell_plus_1: float,
    kappa_down: float,
    phase_sensitivity: float = 1.0,
) -> float:
    """Compute phase-locked threshold facilitation from higher level.

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


def update_phase_dynamics(
    phi: float,
    omega: float,
    dt: float,
    coupling_strength: float = 0.0,
    phi_neighbor: float | None = None,
) -> float:
    """Update oscillatory phase with optional coupling.

    dϕ/dt = ω + coupling term

    Args:
        phi: Current phase (radians)
        omega: Natural frequency (rad/ms)
        dt: Time step (ms)
        coupling_strength: Phase coupling to neighbor
        phi_neighbor: Neighboring phase for coupling

    Returns:
        Updated phase (wrapped to [0, 2π])
    """

    dphi = omega * dt

    if phi_neighbor is not None and coupling_strength != 0.0:
        # Kuramoto-style phase coupling
        dphi += coupling_strength * np.sin(phi_neighbor - phi) * dt

    phi_new = (phi + dphi) % (2 * np.pi)

    return float(phi_new)


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
            pi_ell_minus_1 = self.pi[ell - 1] if ell > 0 else None

            dpi_dt = precision_coupling_ode(
                pi_ell=self.pi[ell],
                tau_pi=self.tau_pi,
                epsilon_ell=self.epsilon[ell],
                alpha_gain=alpha_gain,
                pi_ell_plus_1=pi_ell_plus_1,
                pi_ell_minus_1=pi_ell_minus_1,
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
        kappa_down: float = 0.1,
    ) -> np.ndarray:
        """Compute phase-locked thresholds for all levels.

        Args:
            theta_0: Baseline thresholds for each level
            kappa_down: Phase coupling strength

        Returns:
            Phase-modulated thresholds
        """

        thetas = np.zeros(self.n_levels)

        for ell in range(self.n_levels):
            if ell < self.n_levels - 1:
                # Higher level modulates this level
                thetas[ell] = phase_locked_threshold(
                    theta_0_ell=theta_0[ell],
                    pi_ell_plus_1=self.pi[ell + 1],
                    phi_ell_plus_1=self.phi[ell + 1],
                    kappa_down=kappa_down,
                )
            else:
                # Top level: no modulation
                thetas[ell] = theta_0[ell]

        return thetas
