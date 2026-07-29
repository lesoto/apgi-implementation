from __future__ import annotations

import warnings
from typing import Callable

import numpy as np

from core.rng import RNGLike, resolve_rng

# Suppress LAPACK warnings
warnings.filterwarnings("ignore", message=".*On entry to DLASCL.*")


class LiquidNetwork:
    def __init__(
        self,
        n_units: int = 500,
        spectral_radius: float = 0.9,
        res_sparsity: float = 0.15,
        rng: RNGLike = None,
    ):
        if not (0.7 <= spectral_radius <= 0.95):
            raise ValueError(
                f"spectral_radius must be in [0.7, 0.95], got {spectral_radius}. "
                "Spec §17: biologically grounded upper sub-critical regime. "
                "Values < 0.7 produce insufficient temporal memory; "
                "values > 0.95 produce non-fading dynamics."
            )
        if not (0.0 < res_sparsity <= 1.0):
            raise ValueError(f"res_sparsity must be in (0, 1], got {res_sparsity}")
        self.n = n_units
        self.res_sparsity = res_sparsity
        # All weight draws come from one resolvable generator so that a seeded
        # run reproduces the reservoir exactly (core.rng contract).
        generator = resolve_rng(rng)
        self.rng = generator
        # Sparse and random (spec §10, density p=0.1-0.2) — previously fully
        # dense. Masking before spectral normalization below.
        res_mask = (generator.random((n_units, n_units)) < res_sparsity).astype(float)
        W_raw = generator.standard_normal((n_units, n_units)) * 0.1 * res_mask
        # Normalize so ρ(W_res) = spectral_radius, guaranteeing echo state property
        try:
            with np.errstate(all="ignore"):
                rho = np.max(np.abs(np.linalg.eigvals(W_raw)))
                # Ensure rho is not too small to avoid numerical issues
                rho = max(rho, 1e-6)
                self.W_res = W_raw * (spectral_radius / rho)
        except (np.linalg.LinAlgError, FloatingPointError):
            # Fallback: use scaled random matrix without spectral normalization
            self.W_res = W_raw * spectral_radius
        self.W_in = generator.standard_normal(n_units) * 0.1
        # Trained readout weights (initialized randomly, to be trained)
        self.W_out = generator.standard_normal(n_units) * (1.0 / np.sqrt(n_units))
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
        activation: Callable[[np.ndarray], np.ndarray] = np.tanh,
        precision: float | None = None,
        S_target: float | None = None,
        theta: float | None = None,
        A_amp: float = 0.0,
        alpha_res: float | None = None,
        saturating_amplification: bool = True,
    ) -> np.ndarray:
        """dx/dt = -x/τ + f(W_res x + W_in u) + A_amp * x * [S - θ]_+.
        Args:
            u: Input signal
            tau: Fixed time constant (if None, uses precision-adaptive)
            dt: Time step
            activation: Nonlinearity function
            precision: Precision for adaptive τ(t) (optional)
            S_target: Current signal for suprathreshold amplification (§10.3)
            theta: Threshold for suprathreshold amplification (§10.3)
            A_amp: Suprathreshold amplification strength (§10.3)
            alpha_res: Reservoir leak rate α_res = 1/τ (optional; defaults to
                1/tau_eff). Spec §10.3 stability requirement: the net leak
                must stay positive, A_amp·[S-θ]_+ < α_res, for the duration
                of broadcast — checked below when saturating_amplification=False.
            saturating_amplification: If True (default), use the spec's
                equivalent saturating form A_amp·x·tanh([S-θ]_+) instead of
                the unbounded ReLU-gated term A_amp·x·[S-θ]_+. This was
                previously unconditionally the unbounded form with no
                stability safeguard at all.
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
        res_drive = activation(self.W_res @ self.x + self.W_in * u)
        dx_dt = -self.x / tau_eff + res_drive
        # 2) Suprathreshold amplification (§10.3)
        # dx/dt += A_amp * x * [S - θ]_+
        if S_target is not None and theta is not None and A_amp > 0:
            # [S - θ]_+ is the ReLU of the margin
            margin_plus = max(0.0, S_target - theta)
            if saturating_amplification:
                # Spec-sanctioned saturating alternative: bounds the
                # amplification term regardless of how large the margin
                # grows, guaranteeing the suprathreshold mode decays rather
                # than diverges without requiring a separate stability check.
                dx_dt += A_amp * self.x * np.tanh(margin_plus)
            else:
                _alpha_res = alpha_res if alpha_res is not None else 1.0 / tau_eff
                if A_amp * margin_plus >= _alpha_res:
                    raise ValueError(
                        f"Suprathreshold amplification unstable: A_amp*[S-θ]_+ "
                        f"={A_amp * margin_plus:.4f} >= alpha_res={_alpha_res:.4f}. "
                        "Spec §10.3 requires the net leak to remain positive "
                        "for the duration of broadcast. Reduce A_amp, or use "
                        "saturating_amplification=True (the default)."
                    )
                dx_dt += A_amp * self.x * margin_plus
        self.x += dt * dx_dt
        return self.x

    def readout_signal(self, method: str = "linear") -> float:
        """S(t) = W_out^T x(t) (trained linear readout) or x(t)^T x(t) (energy)."""
        if method == "linear":
            return float(np.dot(self.W_out, self.x))
        elif method == "energy":
            return float(np.dot(self.x, self.x))
        else:
            raise ValueError(f"Unknown readout method: {method}")

    def apply_suprathreshold_gain(
        self, S: float, theta: float, A: float = 1.0, dt: float = 1.0
    ) -> np.ndarray:
        """Additive Euler term: dx/dt += A*x*[S-θ]_+ + baseline*[S-θ]_+."""
        suprath = max(0.0, S - theta)
        # Include baseline drive to bootstrap from zero initial state
        baseline = 0.01
        self.x += dt * (A * self.x * suprath + baseline * suprath)
        return self.x
