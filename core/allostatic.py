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
) -> float:
    """Compute dθ/dt for allostatic threshold dynamics per APGI spec.

    Formula: dθ/dt = -γ_θ(θ - θ_base) + δ_reset·B(t) + η[C(t) - V(t)]

    Components:
    - -γ_θ(θ - θ_base): Mean-reversion to baseline (exponential decay)
    - δ_reset·B(t): Post-ignition refractory boost
    - η[C(t) - V(t)]: Allostatic cost-value mismatch

    Args:
        theta: Current threshold
        theta_0: Baseline threshold (θ_base)
        gamma: Mean-reversion rate (γ_θ = 1/τ_θ)
        B_prev: Previous ignition state (0 or 1)
        delta: Refractory boost magnitude (δ_reset)

    Returns:
        dθ/dt (threshold change rate)
    """

    mean_reversion = -gamma * (theta - theta_0)
    refractory = delta * B_prev

    return float(mean_reversion + refractory)


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

    def step(self, C: float, V: float, eta: float, B: int) -> float:
        """Single ODE step for threshold update.

        Args:
            C: Metabolic cost
            V: Information value
            eta: Allostatic learning rate
            B: Current ignition state

        Returns:
            Updated threshold
        """

        # Compute threshold ODE per APGI spec
        # dθ/dt = -γ_θ(θ - θ_base) + δ_reset·B(t) + η[C(t) - V(t)]
        mean_reversion = -self.gamma * (self.theta - self.theta_0)
        refractory = self.delta * self.B_prev
        allostatic = eta * (C - V)

        theta_dot = mean_reversion + refractory + allostatic

        # Euler update
        self.theta = update_threshold_euler(self.theta, theta_dot, self.dt)

        # Store state for next step
        self.B_prev = B

        return self.theta

    def reset(self, theta: float | None = None):
        """Reset controller state."""

        self.theta = theta if theta is not None else self.theta_0
        self.B_prev = 0
