from __future__ import annotations

import numpy as np


class LiquidNetwork:
    def __init__(self, n_units: int = 500, spectral_radius: float = 0.9):
        if not (0.0 < spectral_radius < 1.0):
            raise ValueError(
                "spectral_radius must be in (0, 1) for echo state property"
            )
        self.n = n_units
        W_raw = np.random.randn(n_units, n_units) * 0.1
        # Normalize so ρ(W_res) = spectral_radius, guaranteeing echo state property
        rho = np.max(np.abs(np.linalg.eigvals(W_raw)))
        self.W_res = W_raw * (spectral_radius / rho) if rho > 0.0 else W_raw
        self.W_in = np.random.randn(n_units) * 0.1
        self.x = np.zeros(n_units, dtype=float)
        # Adaptive time constant state
        self.tau_current = 100.0  # Default τ in ms
        self.tau_base = 100.0

    def compute_adaptive_tau(
        self,
        precision: float,
        tau_min: float = 10.0,
        tau_max: float = 500.0,
        pi_ref: float = 1.0,
    ) -> float:
        """Compute adaptive time constant τ(t) modulated by precision.

        Higher precision → faster dynamics (lower τ)
        Lower precision → slower dynamics (higher τ)

        Formula: τ(t) = τ_base · (π_ref / π)^α with α=0.5

        Args:
            precision: Current precision value
            tau_min: Minimum time constant (fastest)
            tau_max: Maximum time constant (slowest)
            pi_ref: Reference precision for baseline

        Returns:
            Adaptive time constant τ(t)
        """

        if precision <= 0:
            return tau_max

        # Inverse relationship: high precision → fast (low tau)
        alpha = 0.5  # Modulation exponent
        tau_adaptive = self.tau_base * (pi_ref / precision) ** alpha

        return float(np.clip(tau_adaptive, tau_min, tau_max))

    def step(
        self,
        u: float,
        tau: float | None = None,
        dt: float = 1.0,
        activation=np.tanh,
        precision: float | None = None,
        S_target: float | None = None,
        theta: float | None = None,
        A_amp: float = 0.0,
    ):
        """x_dot = -x/τ + f(W_res x + W_in u) + A_amp * x * [S - θ]_+.

        Args:
            u: Input signal
            tau: Fixed time constant (if None, uses precision-adaptive)
            dt: Time step
            activation: Nonlinearity function
            precision: Precision for adaptive τ(t) (optional)
            S_target: Current signal for suprathreshold amplification (§10.3)
            theta: Threshold for suprathreshold amplification (§10.3)
            A_amp: Suprathreshold amplification strength
        """

        # Determine time constant
        if tau is not None:
            tau_eff = tau
        elif precision is not None:
            tau_eff = self.compute_adaptive_tau(precision)
            self.tau_current = tau_eff
        else:
            tau_eff = self.tau_current

        if tau_eff <= 0:
            raise ValueError("tau must be > 0")

        # 1) Standard reservoir dynamics (§10.1)
        # dx/dt = -x/τ + f(W_res x + W_in u)
        dx_dt = -self.x / tau_eff + activation(self.W_res @ self.x + self.W_in * u)

        # 2) Suprathreshold amplification (§10.3)
        # dx/dt += A_amp * x * [S - θ]_+
        if S_target is not None and theta is not None and A_amp > 0:
            suprath_gain = max(0.0, S_target - theta)
            dx_dt += A_amp * self.x * suprath_gain

        self.x += dt * dx_dt
        return self.x

    def readout_signal(self) -> float:
        """S(t) = x(t)^T x(t)."""

        return float(np.dot(self.x, self.x))

    def apply_suprathreshold_gain(
        self, S: float, theta: float, A: float = 1.0, dt: float = 1.0
    ):
        """Additive Euler term: dx/dt += A*x*[S-θ]_+."""

        suprath = max(0.0, S - theta)
        self.x += dt * (A * self.x * suprath)
        return self.x
