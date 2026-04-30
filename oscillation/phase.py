"""Oscillatory phase dynamics for APGI.

Implements explicit phase state ϕ_l(t) = ω_l t + ϕ_0 with cross-level coupling.
"""

from __future__ import annotations

import numpy as np


def compute_phase(t: float, omega: float, phi_0: float = 0.0) -> float:
    """Compute oscillatory phase at time t.

    Formula: ϕ_l(t) = ω_l · t + ϕ_0

    Args:
        t: Time (seconds or ms depending on context)
        omega: Angular frequency (rad/time unit)
        phi_0: Initial phase (radians)

    Returns:
        Phase ϕ_l(t) wrapped to [0, 2π]
    """

    phi = (omega * t + phi_0) % (2 * np.pi)
    return float(phi)


def update_phase_euler(
    phi: float,
    omega: float,
    dt: float,
    coupling_sum: float = 0.0,
) -> float:
    """Update phase using Euler integration.

    dϕ/dt = ω + coupling

    Args:
        phi: Current phase (radians)
        omega: Natural frequency (rad/time)
        dt: Time step
        coupling_sum: Sum of coupling terms from other oscillators

    Returns:
        Updated phase (wrapped to [0, 2π])
    """

    dphi = (omega + coupling_sum) * dt
    phi_new = (phi + dphi) % (2 * np.pi)
    return float(phi_new)


def phase_coupling_kuramoto(
    phi_i: float,
    phi_j: float,
    K_ij: float,
    omega_i: float,
) -> float:
    """Kuramoto phase coupling between two oscillators.

    Coupling term: K_ij · sin(ϕ_j - ϕ_i)

    This is the standard Kuramoto model coupling that drives phase
    synchronization between oscillators.

    Args:
        phi_i: Phase of receiving oscillator i (radians)
        phi_j: Phase of sending oscillator j (radians)
        K_ij: Coupling strength from j to i
        omega_i: Natural frequency of oscillator i (unused but kept for API consistency)

    Returns:
        Coupling term contribution to dϕ_i/dt
    """

    return float(K_ij * np.sin(phi_j - phi_i))


def hierarchical_phase_coupling(
    phases: np.ndarray,
    omegas: np.ndarray,
    K_matrix: np.ndarray,
) -> np.ndarray:
    """Compute phase coupling for hierarchical oscillator array.

    For a hierarchy of oscillators (typically one per level),
    computes coupling from adjacent levels.

    Args:
        phases: Phase array [ϕ_0, ϕ_1, ..., ϕ_{L-1}] for L levels
        omegas: Natural frequencies for each level
        K_matrix: Coupling matrix K[i,j] = strength from j to i

    Returns:
        Array of coupling sums for each oscillator
    """

    n = len(phases)
    coupling_sums = np.zeros(n)

    for i in range(n):
        for j in range(n):
            if K_matrix[i, j] != 0.0:
                coupling_sums[i] += phase_coupling_kuramoto(
                    phases[i], phases[j], K_matrix[i, j], omegas[i]
                )

    return coupling_sums


def nearest_neighbor_coupling_matrix(
    n_levels: int,
    K_up: float,
    K_down: float,
) -> np.ndarray:
    """Create coupling matrix for nearest-neighbor hierarchical coupling.

    Each level couples to adjacent levels with:
    - K_down: coupling from higher level (i+1 to i)
    - K_up: coupling from lower level (i-1 to i)

    Args:
        n_levels: Number of hierarchy levels
        K_up: Upward coupling strength (lower → higher)
        K_down: Downward coupling strength (higher → lower)

    Returns:
        K_matrix for hierarchical_phase_coupling
    """

    K = np.zeros((n_levels, n_levels))

    for i in range(n_levels):
        if i > 0:
            K[i, i - 1] = K_up  # Coupling from lower level
        if i < n_levels - 1:
            K[i, i + 1] = K_down  # Coupling from higher level

    return K


class PhaseOscillatorNetwork:
    """Network of coupled phase oscillators for hierarchical APGI."""

    def __init__(
        self,
        n_levels: int,
        frequencies: np.ndarray | None = None,
        coupling_matrix: np.ndarray | None = None,
    ):
        """Initialize phase oscillator network.

        Args:
            n_levels: Number of oscillators (typically equals hierarchy levels)
            frequencies: Natural frequencies (Hz). If None, uses log-spaced.
            coupling_matrix: Coupling matrix K[i,j]. If None, uses nearest-neighbor.
        """

        self.n_levels = n_levels

        if frequencies is None:
            # Default: log-spaced from 1 Hz to ~100 Hz
            self.omegas = 2 * np.pi * np.logspace(0, 2, n_levels)
        else:
            self.omegas = 2 * np.pi * np.array(frequencies)

        if coupling_matrix is None:
            self.K = nearest_neighbor_coupling_matrix(n_levels, 0.1, 0.2)
        else:
            self.K = np.array(coupling_matrix)

        # State
        self.phases = np.zeros(n_levels)  # Initial phases
        self.t = 0.0

    def step(self, dt: float = 1.0) -> np.ndarray:
        """Update all oscillator phases by one time step.

        Args:
            dt: Time step

        Returns:
            Updated phases
        """

        # Compute coupling contributions
        coupling_sums = hierarchical_phase_coupling(self.phases, self.omegas, self.K)

        # Update each phase
        for i in range(self.n_levels):
            self.phases[i] = update_phase_euler(
                self.phases[i], self.omegas[i], dt, coupling_sums[i]
            )

        self.t += dt
        return self.phases.copy()

    def get_phases(self) -> np.ndarray:
        """Get current phases."""
        return self.phases.copy()

    def set_phases(self, phases: np.ndarray) -> None:
        """Set phases manually."""
        self.phases = np.array(phases) % (2 * np.pi)
