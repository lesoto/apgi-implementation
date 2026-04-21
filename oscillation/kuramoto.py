"""Kuramoto oscillators with phase noise and reset dynamics.

Implements coupled phase dynamics per APGI spec §9:
    dφ_ℓ/dt = ω_ℓ + Σ_j K_{ℓj} sin(φ_j - φ_ℓ) + ξ_ℓ(t)

where ξ_ℓ(t) is Ornstein-Uhlenbeck phase noise.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class OrnsteinUhlenbeckNoise:
    """Ornstein-Uhlenbeck noise process for phase dynamics.

    Implements: dξ = -ξ/τ_ξ dt + σ_ξ dW

    This provides colored noise with exponential autocorrelation,
    more realistic than white noise for neural oscillations.
    """

    tau_xi: float = 1.0  # Correlation timescale (ms)
    sigma_xi: float = 0.1  # Noise amplitude (rad/ms)

    def __post_init__(self):
        self.xi = 0.0  # Current noise state

    def step(self, dt: float) -> float:
        """Update OU noise process.

        Args:
            dt: Time step

        Returns:
            Current noise value ξ(t)
        """
        # OU update: ξ(t+dt) = ξ(t) exp(-dt/τ_ξ) + σ_ξ √(1-exp(-2dt/τ_ξ)) N(0,1)
        decay = np.exp(-dt / self.tau_xi)
        diffusion = self.sigma_xi * np.sqrt(1.0 - decay**2)

        self.xi = decay * self.xi + diffusion * np.random.normal()
        return float(self.xi)

    def reset(self):
        """Reset noise to zero."""
        self.xi = 0.0


class KuramotoOscillators:
    """Coupled Kuramoto oscillators with phase noise and reset.

    Implements the coupled phase dynamics:
        dφ_ℓ/dt = ω_ℓ + Σ_j K_{ℓj} sin(φ_j - φ_ℓ) + ξ_ℓ(t)

    where:
    - ω_ℓ: natural frequency of oscillator ℓ
    - K_{ℓj}: coupling strength from j to ℓ
    - ξ_ℓ(t): Ornstein-Uhlenbeck phase noise

    Spec §9: Oscillatory Phase Coupling
    """

    def __init__(
        self,
        n_levels: int,
        frequencies: np.ndarray | None = None,
        coupling_matrix: np.ndarray | None = None,
        tau_xi: float = 1.0,
        sigma_xi: float = 0.1,
    ):
        """Initialize Kuramoto oscillator network.

        Args:
            n_levels: Number of oscillators (typically = hierarchy levels)
            frequencies: Natural frequencies (Hz). If None, uses log-spaced.
            coupling_matrix: Coupling matrix K[i,j]. If None, uses nearest-neighbor.
            tau_xi: OU noise correlation timescale (ms)
            sigma_xi: OU noise amplitude (rad/ms)
        """

        self.n_levels = n_levels

        # Natural frequencies (convert Hz to rad/ms)
        if frequencies is None:
            # Default: log-spaced from 1 Hz to ~100 Hz
            freqs_hz = np.logspace(0, 2, n_levels)
            self.omegas = 2 * np.pi * freqs_hz / 1000.0  # Convert to rad/ms
        else:
            self.omegas = 2 * np.pi * np.array(frequencies) / 1000.0

        # Coupling matrix
        if coupling_matrix is None:
            self.K = self._nearest_neighbor_coupling(n_levels)
        else:
            self.K = np.array(coupling_matrix)

        # Phase state
        self.phases = np.random.uniform(0, 2 * np.pi, n_levels)

        # Phase noise processes (one per oscillator)
        self.noise_processes = [
            OrnsteinUhlenbeckNoise(tau_xi=tau_xi, sigma_xi=sigma_xi)
            for _ in range(n_levels)
        ]

        self.t = 0.0
        self.history: list[np.ndarray] = []

    @staticmethod
    def _nearest_neighbor_coupling(n_levels: int) -> np.ndarray:
        """Create nearest-neighbor coupling matrix.

        Each level couples to adjacent levels with:
        - K_down = 0.2: coupling from higher level (i+1 to i)
        - K_up = 0.1: coupling from lower level (i-1 to i)

        Args:
            n_levels: Number of levels

        Returns:
            Coupling matrix K[i,j] = strength from j to i
        """
        K = np.zeros((n_levels, n_levels))
        K_up = 0.1
        K_down = 0.2

        for i in range(n_levels):
            if i > 0:
                K[i, i - 1] = K_up  # Coupling from lower level
            if i < n_levels - 1:
                K[i, i + 1] = K_down  # Coupling from higher level

        return K

    def step(self, dt: float = 1.0) -> np.ndarray:
        """Update all oscillator phases by one time step.

        Implements Euler-Maruyama integration of:
            dφ_ℓ/dt = ω_ℓ + Σ_j K_{ℓj} sin(φ_j - φ_ℓ) + ξ_ℓ(t)

        Args:
            dt: Time step (ms)

        Returns:
            Updated phases (wrapped to [0, 2π])
        """

        # Compute coupling contributions
        coupling_sums = np.zeros(self.n_levels)
        for i in range(self.n_levels):
            for j in range(self.n_levels):
                if self.K[i, j] != 0.0:
                    # Kuramoto coupling: K_ij sin(φ_j - φ_i)
                    coupling_sums[i] += self.K[i, j] * np.sin(
                        self.phases[j] - self.phases[i]
                    )

        # Update each phase with noise
        for i in range(self.n_levels):
            # Get OU noise for this oscillator
            noise = self.noise_processes[i].step(dt)

            # Phase update: dφ = (ω + coupling + noise) dt
            dphi = (self.omegas[i] + coupling_sums[i] + noise) * dt
            self.phases[i] = (self.phases[i] + dphi) % (2 * np.pi)

        self.t += dt
        self.history.append(self.phases.copy())

        return self.phases.copy()

    def reset_phase_on_ignition(
        self,
        level: int,
        reset_amount: float = np.pi,
    ) -> None:
        """Reset phase on ignition event.

        Spec §9: Phase reset on ignition

        When ignition occurs at level ℓ, reset its phase by π radians
        to represent the "reset" of neural activity.

        Args:
            level: Level index where ignition occurred
            reset_amount: Amount to reset phase (default π)
        """

        if 0 <= level < self.n_levels:
            self.phases[level] = (self.phases[level] + reset_amount) % (2 * np.pi)
            # Also reset the noise process
            self.noise_processes[level].reset()

    def get_phases(self) -> np.ndarray:
        """Get current phases."""
        return self.phases.copy()

    def set_phases(self, phases: np.ndarray) -> None:
        """Set phases manually."""
        self.phases = np.array(phases) % (2 * np.pi)

    def get_synchronization_order(self) -> float:
        """Compute Kuramoto order parameter (synchronization measure).

        R = |Σ_ℓ exp(i φ_ℓ)| / n_levels

        Returns:
            Synchronization order (0 = incoherent, 1 = fully synchronized)
        """

        complex_sum = np.sum(np.exp(1j * self.phases))
        R = np.abs(complex_sum) / self.n_levels
        return float(R)

    def get_phase_coherence(self) -> np.ndarray:
        """Compute pairwise phase coherence between all oscillators.

        Returns:
            Coherence matrix C[i,j] = |cos(φ_i - φ_j)|
        """

        coherence = np.zeros((self.n_levels, self.n_levels))
        for i in range(self.n_levels):
            for j in range(self.n_levels):
                phase_diff = self.phases[i] - self.phases[j]
                coherence[i, j] = np.abs(np.cos(phase_diff))

        return coherence

    def get_history(self) -> np.ndarray:
        """Get phase history as array (T, n_levels)."""
        if not self.history:
            return np.array([self.phases])
        return np.array(self.history)


class HierarchicalKuramotoSystem:
    """Kuramoto oscillators integrated with hierarchical APGI system.

    Couples oscillator phases to:
    - Threshold modulation (phase-gated ignition)
    - Signal accumulation (phase-weighted precision)
    - Cross-level coupling (top-down/bottom-up)
    """

    def __init__(
        self,
        n_levels: int,
        config: dict | None = None,
    ):
        """Initialize hierarchical Kuramoto system.

        Args:
            n_levels: Number of hierarchy levels
            config: Configuration dict with Kuramoto parameters
        """

        self.n_levels = n_levels
        self.config = config or {}

        # Initialize Kuramoto oscillators
        tau_xi = self.config.get("kuramoto_tau_xi", 1.0)
        sigma_xi = self.config.get("kuramoto_sigma_xi", 0.1)

        self.oscillators = KuramotoOscillators(
            n_levels=n_levels,
            tau_xi=tau_xi,
            sigma_xi=sigma_xi,
        )

    def step(self, dt: float = 1.0) -> dict:
        """Update Kuramoto system and return diagnostics.

        Args:
            dt: Time step

        Returns:
            Dictionary with phases, synchronization, coherence
        """

        phases = self.oscillators.step(dt)

        return {
            "phases": phases,
            "synchronization": self.oscillators.get_synchronization_order(),
            "coherence": self.oscillators.get_phase_coherence(),
        }

    def apply_ignition_reset(self, level: int) -> None:
        """Apply phase reset on ignition at given level.

        Args:
            level: Level where ignition occurred
        """

        reset_amount = self.config.get("kuramoto_reset_amount", np.pi)
        self.oscillators.reset_phase_on_ignition(level, reset_amount)

    def get_phase_modulation_factor(self, level: int) -> float:
        """Get phase-dependent modulation factor for threshold.

        Uses cos(φ_ℓ) to modulate threshold:
        - When φ_ℓ ≈ 0: cos(φ) ≈ 1 (high threshold)
        - When φ_ℓ ≈ π: cos(φ) ≈ -1 (low threshold)

        Args:
            level: Level index

        Returns:
            Modulation factor in [-1, 1]
        """

        if 0 <= level < self.n_levels:
            return float(np.cos(self.oscillators.phases[level]))
        return 0.0
