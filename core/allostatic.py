"""Allostatic threshold dynamics with urgency sensitivity.

Implements continuous-time ODE for dynamic threshold adaptation:
dθ/dt = γ(θ_0 - θ) + δ·B_{t-1} - λ·|dS/dt|
"""

from __future__ import annotations

import numpy as np


def compute_signal_derivative(
    S_current: float,
    S_prev: float,
    dt: float = 1.0,
) -> float:
    """Compute time derivative of signal dS/dt.

    Args:
        S_current: Current signal value
        S_prev: Previous signal value
        dt: Time step

    Returns:
        dS/dt (change rate)
    """

    return float((S_current - S_prev) / dt)


def allostatic_threshold_ode(
    theta: float,
    theta_0: float,
    gamma: float,
    B_prev: int,
    delta: float,
    dS_dt: float,
    lambda_urgency: float,
) -> float:
    """Compute dθ/dt for allostatic threshold dynamics.

    Formula: dθ/dt = γ(θ_0 - θ) + δ·B_{t-1} - λ·|dS/dt|

    Components:
    - γ(θ_0 - θ): Baseline attraction (homeostatic pull)
    - δ·B_{t-1}: Post-ignition refractory boost
    - -λ·|dS/dt|: Urgency sensitivity (faster signal → lower threshold)

    Args:
        theta: Current threshold
        theta_0: Baseline threshold
        gamma: Baseline attraction rate
        B_prev: Previous ignition state (0 or 1)
        delta: Refractory boost magnitude
        dS_dt: Signal derivative (rate of change)
        lambda_urgency: Urgency sensitivity coefficient

    Returns:
        dθ/dt (threshold change rate)
    """

    homeostatic = gamma * (theta_0 - theta)
    refractory = delta * B_prev
    urgency = -lambda_urgency * abs(dS_dt)

    return float(homeostatic + refractory + urgency)


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


def compute_urgency_factor(
    dS_dt: float,
    lambda_urgency: float = 0.1,
    normalize: bool = True,
) -> float:
    """Compute urgency-based threshold modulation factor.

    Urgency reduces threshold when signal changes rapidly, enabling
    faster response to unexpected events.

    Args:
        dS_dt: Signal rate of change
        lambda_urgency: Urgency sensitivity
        normalize: Whether to apply tanh normalization

    Returns:
        Urgency factor (typically negative, reducing threshold)
    """

    raw_urgency = -lambda_urgency * abs(dS_dt)

    if normalize:
        # Soft normalization to prevent extreme values
        return float(np.tanh(raw_urgency))

    return float(raw_urgency)


class AllostaticThresholdController:
    """Continuous-time threshold controller with ODE dynamics."""

    def __init__(
        self,
        theta_0: float = 1.0,
        gamma: float = 0.01,
        delta: float = 0.5,
        lambda_urgency: float = 0.1,
        dt: float = 1.0,
    ):
        """Initialize allostatic controller.

        Args:
            theta_0: Baseline threshold
            gamma: Homeostatic attraction rate
            delta: Refractory boost magnitude
            lambda_urgency: Urgency sensitivity
            dt: Integration time step
        """

        self.theta = theta_0
        self.theta_0 = theta_0
        self.gamma = gamma
        self.delta = delta
        self.lambda_urgency = lambda_urgency
        self.dt = dt
        self.S_prev = 0.0
        self.B_prev = 0

    def step(self, S: float, B: int) -> float:
        """Single ODE step for threshold update.

        Args:
            S: Current accumulated signal
            B: Current ignition state

        Returns:
            Updated threshold
        """

        # Compute signal derivative
        dS_dt = compute_signal_derivative(S, self.S_prev, self.dt)

        # Compute threshold ODE
        theta_dot = allostatic_threshold_ode(
            theta=self.theta,
            theta_0=self.theta_0,
            gamma=self.gamma,
            B_prev=self.B_prev,
            delta=self.delta,
            dS_dt=dS_dt,
            lambda_urgency=self.lambda_urgency,
        )

        # Euler update
        self.theta = update_threshold_euler(self.theta, theta_dot, self.dt)

        # Store state for next step
        self.S_prev = S
        self.B_prev = B

        return self.theta

    def reset(self, theta: float | None = None):
        """Reset controller state."""

        self.theta = theta if theta is not None else self.theta_0
        self.S_prev = 0.0
        self.B_prev = 0
