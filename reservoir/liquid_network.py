from __future__ import annotations

import numpy as np


class LiquidNetwork:
    def __init__(self, n_units: int = 500, spectral_radius: float = 0.9):
        if not (0.0 < spectral_radius < 1.0):
            raise ValueError("spectral_radius must be in (0, 1) for echo state property")
        self.n = n_units
        W_raw = np.random.randn(n_units, n_units) * 0.1
        # Normalize so ρ(W_res) = spectral_radius, guaranteeing echo state property
        rho = np.max(np.abs(np.linalg.eigvals(W_raw)))
        self.W_res = W_raw * (spectral_radius / rho) if rho > 0.0 else W_raw
        self.W_in = np.random.randn(n_units) * 0.1
        self.x = np.zeros(n_units, dtype=float)

    def step(self, u: float, tau: float, dt: float = 1.0, activation=np.tanh):
        """x_dot = -x/τ + f(W_res x + W_in u)."""

        if tau <= 0:
            raise ValueError("tau must be > 0")
        dx_dt = -self.x / tau + activation(self.W_res @ self.x + self.W_in * u)
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
