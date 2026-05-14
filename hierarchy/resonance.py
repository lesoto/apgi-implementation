"""Cross-Level Threshold Resonance — Russian Doll Architecture.

Implements the nested resonance system from §8 / §9 of the APGI Full Specs:

  θ_l[t] = θ_{0,l} · [1 + κ_down · Π_{l+1} · cos(φ_{l+1})]   (§8 spec formula)

  dφ_l/dt = ω_l + κ_down · sin(φ_{l+1} − φ_l)                 (top-down entrainment)

  S_l[t+1] = (1 − λ_l) · S_l[t] + λ_l · S_inst_l[t]          (per-level accumulator)

Each level l owns:
  φ_l  — oscillatory phase (advances at natural frequency ω_l = 2π / τ_l)
  θ_l  — threshold continuously re-shaped by the cosine of the level above
  S_l  — ignition signal accumulated from level-specific salience

Higher levels oscillate more slowly (large τ_l).  The top-down phase-coupling
term sin(φ_{l+1} − φ_l) pulls each level toward its superior's phase, creating
nested rhythmic windows of ignition opportunity — the Russian Doll property.

The primary ignition check (level 0, sensory) uses θ_0[t], which is
continuously modulated by level 1's phase and precision.  Level 1's θ_1 is
modulated by level 2's phase, etc., nesting the hierarchy.

When an ignition event fires at level 0, a post-ignition refractory boost
δ_refractory is added to θ_0 and the level-0 signal is partially reset (ρ_S),
matching §17.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LevelState:
    """State variables owned by one level in the resonance system."""

    S: float  # accumulated ignition signal
    theta: float  # phase-modulated threshold (updated each step)
    phi: float  # oscillatory phase [0, 2π)
    pi: float  # precision (passed in from the precision pipeline)


class NestedResonanceSystem:
    """Per-level phase-modulated threshold resonance (Russian Doll architecture).

    Parameters
    ----------
    n_levels : int
        Number of cortical hierarchy levels L.
    theta_0 : array-like, shape (L,)
        Baseline thresholds for each level.
    omega : array-like, shape (L,)
        Natural angular frequencies ω_l = 2π / τ_l.
    lambda_rates : array-like, shape (L,)
        Per-level leaky-accumulator rates λ_l = dt / τ_l (clamped to (0,1)).
    kappa_down : float
        Top-down coupling strength κ_down (both phase entrainment and
        threshold modulation use the same parameter per spec §8).
    kappa_up : float
        Bottom-up suppression strength (post-ignition cascade).
    theta_min, theta_max : float
        Hard clamps on modulated thresholds.
    phi_noise_std : float
        Per-step Gaussian noise on phases (biological phase jitter).
    rng : numpy Generator | None
        Random number generator for reproducible phase noise.
    """

    def __init__(
        self,
        n_levels: int,
        theta_0: np.ndarray,
        omega: np.ndarray,
        lambda_rates: np.ndarray,
        kappa_down: float = 0.1,
        kappa_up: float = 0.0,
        theta_min: float = 0.1,
        theta_max: float = 20.0,
        phi_noise_std: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        if n_levels < 1:
            raise ValueError("n_levels must be >= 1")

        self.n_levels = n_levels
        self.theta_0 = np.asarray(theta_0, dtype=float)
        self.omega = np.asarray(omega, dtype=float)
        self.lambda_rates = np.clip(np.asarray(lambda_rates, dtype=float), 1e-6, 1.0 - 1e-6)
        self.kappa_down = float(kappa_down)
        self.kappa_up = float(kappa_up)
        self.theta_min = float(theta_min)
        self.theta_max = float(theta_max)
        self.phi_noise_std = float(phi_noise_std)
        self.rng = rng or np.random.default_rng()

        # Per-level state
        self.S = np.zeros(n_levels)
        self.phi = np.zeros(n_levels)
        self.pi = np.ones(n_levels)
        # Initialise thresholds at their baselines
        self.theta = self.theta_0.copy()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def step(
        self,
        S_inst_levels: np.ndarray,
        pi_levels: np.ndarray,
        dt: float,
    ) -> None:
        """Advance the full resonance system by one timestep.

        Execution order (matches §20 pipeline steps 9-14):
          1. Update per-level precisions from the precision pipeline.
          2. Advance oscillatory phases with top-down Kuramoto entrainment.
          3. Recompute per-level thresholds via the spec PAC formula.
          4. Accumulate per-level ignition signals (leaky integrators).

        Parameters
        ----------
        S_inst_levels : array-like (L,)
            Per-level instantaneous salience:
            S_inst_l = Π_l · φ(ε_l) · Γ_l
        pi_levels : array-like (L,)
            Per-level precisions from the precision pipeline.
        dt : float
            Integration timestep.
        """
        S_inst = np.asarray(S_inst_levels, dtype=float)
        pi = np.asarray(pi_levels, dtype=float)
        self.pi = pi.copy()

        # 1 — Phase advancement with top-down Kuramoto entrainment
        self.phi = self._advance_phases(dt)

        # 2 — Threshold update using the spec formula:
        #     θ_l = θ_{0,l} · (1 + κ_down · Π_{l+1} · cos(φ_{l+1}))
        self.theta = self._compute_thresholds()

        # 3 — Per-level leaky accumulation:
        #     S_l(t+1) = (1 − λ_l) S_l + λ_l S_inst_l
        self.S = (1.0 - self.lambda_rates) * self.S + self.lambda_rates * S_inst

    def apply_level_ignition(
        self,
        level: int,
        rho_S: float = 0.1,
        delta_refractory: float = 0.5,
    ) -> None:
        """Apply post-ignition refractory reset for the given level (§17).

        Parameters
        ----------
        level : int
            The level that fired (usually 0 for the primary ignition).
        rho_S : float
            Signal retention fraction after reset (ρ_S ∈ [0.1, 0.5]).
        delta_refractory : float
            Temporary threshold elevation δ_refractory.
        """
        if not (0 <= level < self.n_levels):
            raise ValueError(f"level {level} out of range [0, {self.n_levels})")
        self.S[level] *= rho_S
        self.theta[level] = min(self.theta[level] + delta_refractory, self.theta_max)
        # Bottom-up cascade: lower-level ignition suppresses the level above
        if level + 1 < self.n_levels and self.kappa_up > 0.0:
            self.theta[level + 1] *= 1.0 - self.kappa_up

    @property
    def primary_signal(self) -> float:
        """Accumulated ignition signal at the bottom level (primary sensory)."""
        return float(self.S[0])

    @property
    def primary_threshold(self) -> float:
        """Phase-modulated threshold at the bottom level."""
        return float(self.theta[0])

    @property
    def ignition_windows(self) -> np.ndarray:
        """Boolean array: True where S_l exceeds θ_l at the current timestep."""
        return self.S > self.theta

    @property
    def modulation_depth(self) -> np.ndarray:
        """Per-level modulation depth: Δθ_l / θ_{0,l} = κ · Π_{l+1} · cos(φ_{l+1}).

        Zero at the top level; positive when in an inhibitory half-cycle,
        negative when in an excitatory half-cycle.
        """
        depth = np.zeros(self.n_levels)
        for level in range(self.n_levels - 1):
            depth[level] = self.kappa_down * self.pi[level + 1] * np.cos(self.phi[level + 1])
        return depth

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _advance_phases(self, dt: float) -> np.ndarray:
        """Advance φ_l using natural frequency + top-down Kuramoto coupling.

        dφ_l/dt = ω_l + κ_down · sin(φ_{l+1} − φ_l)

        The coupling term pulls each level toward the phase of the level above
        it, creating nested synchrony windows (the Russian Doll property).
        """
        phi_new = np.zeros(self.n_levels)
        for level in range(self.n_levels):
            dphi = self.omega[level] * dt

            # Top-down entrainment from level level+1
            if level < self.n_levels - 1:
                dphi += self.kappa_down * np.sin(self.phi[level + 1] - self.phi[level]) * dt

            # Optional biological phase jitter
            if self.phi_noise_std > 0.0:
                dphi += self.phi_noise_std * self.rng.standard_normal() * np.sqrt(dt)

            phi_new[level] = (self.phi[level] + dphi) % (2.0 * np.pi)
        return phi_new

    def _compute_thresholds(self) -> np.ndarray:
        """Apply the §8 spec formula to all levels.

        θ_l = θ_{0,l} · (1 + κ_down · Π_{l+1} · cos(φ_{l+1}))

        The top level has no level above it, so it stays at its baseline θ_{0,L-1}.
        """
        thetas = np.empty(self.n_levels)
        for level in range(self.n_levels):
            if level < self.n_levels - 1:
                mod = 1.0 + self.kappa_down * self.pi[level + 1] * np.cos(self.phi[level + 1])
                thetas[level] = self.theta_0[level] * mod
            else:
                thetas[level] = self.theta_0[level]
        return np.clip(thetas, self.theta_min, self.theta_max)


def build_resonance_system(
    n_levels: int,
    taus: np.ndarray,
    theta_base: float,
    dt: float,
    kappa_down: float = 0.1,
    kappa_up: float = 0.0,
    theta_min: float = 0.1,
    theta_max: float = 20.0,
    phi_noise_std: float = 0.0,
    rng: np.random.Generator | None = None,
) -> NestedResonanceSystem:
    """Convenience factory: build a NestedResonanceSystem from pipeline timescales.

    Parameters
    ----------
    n_levels : int
    taus : array (L,)  — timescale per level (increasing with level index)
    theta_base : float — common baseline threshold for all levels
    dt : float         — integration step (used to derive λ_l = dt/τ_l)
    """
    theta_0 = np.full(n_levels, theta_base)
    omega = 2.0 * np.pi / taus
    # λ_l = dt / τ_l  — faster levels respond more quickly to salience
    lambda_rates = np.clip(dt / taus, 1e-4, 0.999)
    return NestedResonanceSystem(
        n_levels=n_levels,
        theta_0=theta_0,
        omega=omega,
        lambda_rates=lambda_rates,
        kappa_down=kappa_down,
        kappa_up=kappa_up,
        theta_min=theta_min,
        theta_max=theta_max,
        phi_noise_std=phi_noise_std,
        rng=rng,
    )
