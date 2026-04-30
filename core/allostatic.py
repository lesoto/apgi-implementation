"""Allostatic threshold dynamics.

Implements continuous-time ODE for dynamic threshold adaptation per APGI spec:
dθ/dt = -γ_θ(θ - θ_base) + δ_reset·B(t) + η[C(t) - V(t)]

Note: Derivative coupling to dS/dt was explicitly removed from the spec
as it creates a second-order ODE system that is not required for APGI dynamics.
"""

from __future__ import annotations

import numpy as np


def allostatic_threshold_ode(
    theta: float,
    theta_0: float,
    gamma: float,
    B_prev: int,
    delta: float,
    C: float = 0.0,
    V: float = 0.0,
    eta: float = 0.0,
    dt: float = 1.0,
    noise_std: float = 0.01,
) -> float:
    """Compute dθ/dt for allostatic threshold dynamics per APGI spec (§7.2).

    Formula: dθ/dt = -γ_θ(θ - θ_base) + δ_reset·B(t) + η[C(t) - V(t)]

    Implements Euler-Maruyama integration step:
    θ(t+dt) = θ(t) + dθ/dt * dt + noise_std * sqrt(dt) * N(0,1)

    Components:
    - -γ_θ(θ - θ_base): Mean-reversion to baseline (exponential decay)
    - δ_reset·B(t): Post-ignition refractory boost
    - η[C(t) - V(t)]: Allostatic cost-value mismatch
    - η_θ(t): Stochastic noise term

    Args:
        theta: Current threshold
        theta_0: Baseline threshold (θ_base)
        gamma: Mean-reversion rate (γ_θ = 1/τ_θ)
        B_prev: Previous ignition state (0 or 1)
        delta: Refractory boost magnitude (δ_reset)
        C: Metabolic cost
        V: Information value
        eta: Allostatic learning rate
        dt: Integration time step
        noise_std: Noise amplitude

    Returns:
        Updated threshold θ(t+dt)
    """

    mean_reversion = -gamma * (theta - theta_0)
    refractory = delta * B_prev
    allostatic = eta * (C - V)

    drift = mean_reversion + refractory + allostatic
    noise = float(np.random.normal(0.0, noise_std * np.sqrt(dt)))

    return float(theta + drift * dt + noise)


def update_threshold_euler(
    theta: float,
    theta_dot: float,
    dt: float = 1.0,
    theta_min: float = 0.1,
    theta_max: float = 1e6,
) -> float:
    """Euler integration for threshold update.

    θ(t+dt) = θ(t) + dt · dθ/dt

    Args:
        theta: Current threshold
        theta_dot: Threshold derivative (dθ/dt)
        dt: Time step
        theta_min: Minimum threshold (hard floor)
        theta_max: Maximum threshold (hard ceiling)

    Returns:
        Updated threshold
    """

    theta_new = theta + dt * theta_dot
    return float(np.clip(theta_new, theta_min, theta_max))


class AllostaticThresholdController:
    """Continuous-time threshold controller with ODE dynamics per APGI spec."""

    def __init__(
        self,
        theta_0: float = 1.0,
        gamma: float = 0.01,
        delta: float = 0.5,
        dt: float = 1.0,
    ):
        """Initialize allostatic controller.

        Args:
            theta_0: Baseline threshold (θ_base)
            gamma: Mean-reversion rate (γ_θ = 1/τ_θ)
            delta: Refractory boost magnitude (δ_reset)
            dt: Integration time step
        """

        self.theta = theta_0
        self.theta_0 = theta_0
        self.gamma = gamma
        self.delta = delta
        self.dt = dt
        self.B_prev = 0

    def step(self, C: float, V: float, eta: float, B: int, noise_std: float = 0.01) -> float:
        """Single ODE step for threshold update with stochastic scaling.

        Args:
            C: Metabolic cost
            V: Information value
            eta: Allostatic learning rate
            B: Current ignition state
            noise_std: Threshold noise amplitude

        Returns:
            Updated threshold
        """

        # Compute threshold update per APGI spec (§7.2) with correct scaling
        self.theta = allostatic_threshold_ode(
            theta=self.theta,
            theta_0=self.theta_0,
            gamma=self.gamma,
            B_prev=self.B_prev,
            delta=self.delta,
            C=C,
            V=V,
            eta=eta,
            dt=self.dt,
            noise_std=noise_std,
        )

        # Store state for next step
        self.B_prev = B

        return self.theta

    def reset(self, theta: float | None = None) -> None:
        """Reset controller state."""

        self.theta = theta if theta is not None else self.theta_0
        self.B_prev = 0
