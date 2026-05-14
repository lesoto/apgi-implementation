"""Liquid State Machine (Reservoir Computing) implementation for APGI.

APGI Specification §9 + §10: Reservoir Implementation with Competitive Inhibition

This module implements a biologically plausible reservoir computing layer
that provides signal accumulation and amplification through fixed recurrent
dynamics with trained linear readout.

Key features:
    - Fixed random recurrent weights W_res (not trained)
    - W_inh: PV+ interneuron-mediated divisive normalization (§9, Carandini & Heeger, 2012)
    - Trained linear readout via ridge regression
    - Precision-modulated timescale (CfC-like dynamics)
    - Suprathreshold amplification (§10.3)
    - Spectral radius ρ(W_res) ∈ [0.7, 0.95] (§17 biologically-grounded bounds)

The reservoir implements:
    drive = W_res·x + W_in·u
    f_drive = tanh(drive)                           # nonlinear pre-activation
    r = f_drive / (σ_inh² + W_inh·|f_drive|)       # divisive normalization (§9)
    dx/dt = -x/τ_res + r + A_amp·x·[S-θ]₊          # leaky integrator + amplification

where:
    - x ∈ ℝ^N is the reservoir state
    - W_res is fixed random recurrent weights, ρ(W_res) ∈ [0.7, 0.95]
    - W_inh ∈ ℝ^{N×N} non-negative is the PV+ inhibitory weight matrix
    - σ_inh² is the divisive normalization semi-saturation constant
    - W_in is fixed random input weights
    - A_amp is suprathreshold amplification strength
    - [S-θ]₊ = max(0, S-θ) is the ReLU of ignition margin

References:
    - Maass, W., Natschläger, T., & Markram, H. (2002). "Real-time computing
      without stable states: A new framework for neural computation based on
      perturbations"
    - Carandini, M., & Heeger, D. J. (2012). "Normalization as a canonical
      neural computation." Nature Reviews Neuroscience, 13(1), 51–62.
"""

from __future__ import annotations

import warnings
from typing import Callable, Optional

import numpy as np

# Suppress LAPACK warnings
warnings.filterwarnings("ignore", message=".*On entry to DLASCL.*")


class LiquidStateMachine:
    """Reservoir computing layer per APGI spec §9 + §10.

    Implements fixed recurrent network with PV+ divisive normalization (W_inh)
    and trained linear readout. Supports precision-modulated timescale and
    suprathreshold amplification.

    Attributes:
        N: Reservoir size (number of units)
        M: Input dimension
        tau_res: Base time constant (ms)
        spectral_radius: Spectral radius of W_res, must be in [0.7, 0.95] (§17)
        input_scale: Input weight scaling
        W_res: Fixed recurrent weight matrix (N × N), ρ ∈ [0.7, 0.95]
        W_inh: PV+ inhibitory weight matrix (N × N), non-negative (§9)
        sigma_inh2: Divisive normalization semi-saturation constant (§9)
        W_in: Fixed input weight matrix (N × M)
        W_out: Trained output weight matrix (N × 1)
        x: Current reservoir state (N,)
        history: List of past states for training
    """

    def __init__(
        self,
        N: int = 100,
        M: int = 2,
        tau_res: float = 1.0,
        spectral_radius: float = 0.9,
        input_scale: float = 0.1,
        inh_sparsity: float = 0.1,
        inh_scale: float = 0.2,
        sigma_inh2: float = 0.1,
        seed: Optional[int] = None,
    ):
        """Initialize liquid state machine.

        Args:
            N: Reservoir size (default: 100)
            M: Input dimension (default: 2 for [z_e, z_i])
            tau_res: Base time constant (default: 1.0 ms)
            spectral_radius: Spectral radius of W_res (default: 0.9)
                Must be in [0.7, 0.95] per spec §17 (biologically grounded bounds)
            input_scale: Scaling of input weights (default: 0.1)
            inh_sparsity: Fraction of non-zero entries in W_inh (default: 0.1)
                Mirrors biological cortex connection probability ~10%
            inh_scale: Mean non-zero weight in W_inh (default: 0.2)
            sigma_inh2: Semi-saturation constant for divisive normalization (default: 0.1)
                Corresponds to background inhibitory drive; prevents division by zero
            seed: Random seed for reproducibility (optional)

        Raises:
            ValueError: If spectral_radius not in [0.7, 0.95]
        """
        if not (0.7 <= spectral_radius <= 0.95):
            raise ValueError(
                f"spectral_radius must be in [0.7, 0.95], got {spectral_radius}. "
                "Spec §17: ρ(W_res) ∈ [0.7, 0.95] — upper sub-critical regime "
                "consistent with echo state property and biological plausibility. "
                "Values < 0.7 produce insufficient temporal memory; "
                "values > 0.95 produce non-fading dynamics."
            )

        if N <= 0:
            raise ValueError(f"N must be > 0, got {N}")

        if M <= 0:
            raise ValueError(f"M must be > 0, got {M}")

        if tau_res <= 0:
            raise ValueError(f"tau_res must be > 0, got {tau_res}")

        if input_scale <= 0:
            raise ValueError(f"input_scale must be > 0, got {input_scale}")

        self.N = N
        self.M = M
        self.tau_res = tau_res
        self.spectral_radius = spectral_radius
        self.input_scale = input_scale
        self.sigma_inh2 = sigma_inh2

        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)

        # Initialize fixed random weights
        # Recurrent weights: normalize to desired spectral radius
        W_raw = np.random.randn(N, N)
        try:
            with np.errstate(all="ignore"):
                eigs = np.linalg.eigvals(W_raw)
                rho_raw = np.max(np.abs(eigs))
                # Ensure rho_raw is not too small to avoid numerical issues
                rho_raw = max(rho_raw, 1e-6)
                self.W_res = W_raw / rho_raw * spectral_radius
        except (np.linalg.LinAlgError, FloatingPointError):
            # Fallback: use identity matrix scaled by spectral radius
            self.W_res = np.eye(N) * spectral_radius * 0.1

        # Input weights: random with scaling
        self.W_in = np.random.randn(N, M) * input_scale

        # W_inh: PV+ interneuron-mediated divisive normalization (§9).
        # Non-negative sparse matrix — mirrors biological cortex connectivity
        # probability (~10%) and the non-negative nature of inhibitory drive.
        # Each entry W_inh[i,j] weights how much unit j's activation suppresses
        # unit i through PV+ gain control (Carandini & Heeger, 2012).
        mask = (np.random.rand(N, N) < inh_sparsity).astype(float)
        self.W_inh: np.ndarray = np.abs(np.random.randn(N, N)) * inh_scale * mask

        # Output weights: initialized randomly, trained via ridge regression
        self.W_out = np.zeros((N, 1))

        # Reservoir state
        self.x = np.zeros(N, dtype=float)

        # History for training
        self.history: list[np.ndarray] = []
        self.history_targets: list[float] = []

    def step(
        self,
        u: np.ndarray | float,
        tau: Optional[float] = None,
        dt: float = 1.0,
        activation: Callable = np.tanh,
        precision: Optional[float] = None,
        S_target: Optional[float] = None,
        theta: Optional[float] = None,
        A_amp: float = 0.0,
        use_divisive_normalization: bool = False,
    ) -> np.ndarray:
        """Update reservoir state via Euler integration with divisive normalization.

        Full dynamics (§9 + §10.1 + §10.3):
            drive     = W_res·x + W_in·u
            f_drive   = activation(drive)                         # tanh pre-activation
            r         = f_drive / (σ_inh² + W_inh·|f_drive|)    # divisive norm (§9)
            dx/dt     = -x/τ + r + A_amp·x·[S-θ]₊               # leaky + amplification

        The divisive normalization term W_inh·|f_drive| computes the total
        inhibitory drive from PV+ interneurons onto each unit, implementing
        the Carandini-Heeger gain-control mechanism.  Setting
        use_divisive_normalization=False recovers the pre-§9 dynamics.

        Args:
            u: Input signal (scalar or array of shape (M,))
            tau: Fixed time constant (if None, uses precision-adaptive)
            dt: Integration time step (default: 1.0)
            activation: Nonlinearity function (default: tanh)
            precision: Precision for adaptive τ (optional)
            S_target: Current signal for suprathreshold amplification
            theta: Threshold for suprathreshold amplification
            A_amp: Suprathreshold amplification strength (default: 0.0)
            use_divisive_normalization: Apply W_inh PV+ gain control (default: True)

        Returns:
            Updated reservoir state (N,)

        Raises:
            ValueError: If tau ≤ 0 or input shape invalid

        Examples:
            >>> lsm = LiquidStateMachine(N=100, M=2)
            >>> u = np.array([0.5, -0.3])
            >>> x = lsm.step(u, tau=1.0, dt=0.1)
            >>> print(x.shape)
            (100,)

            >>> # With suprathreshold amplification
            >>> x = lsm.step(u, tau=1.0, dt=0.1, S_target=1.5, theta=1.0, A_amp=0.1)
        """
        # Convert scalar input to array
        if np.isscalar(u):
            u = np.array([u])
        else:
            u = np.asarray(u)

        # Handle input validation: exact match required, but allow scalar broadcasting
        if u.shape[0] != self.M:
            # Allow scalar broadcasting: single value can be broadcast to any M
            if u.shape[0] == 1:
                u = np.broadcast_to(u, (self.M,))
            else:
                raise ValueError(f"Input dimension mismatch: expected {self.M}, got {u.shape[0]}")

        # Determine effective time constant
        if tau is not None:
            tau_eff = tau
        elif precision is not None:
            tau_eff = self._compute_adaptive_tau(precision)
        else:
            tau_eff = self.tau_res

        if tau_eff <= 0:
            raise ValueError(f"tau must be > 0, got {tau_eff}")

        # 1) Pre-activation: f(W_res·x + W_in·u)
        recurrent_input = self.W_res @ self.x
        external_input = self.W_in @ u
        f_drive = activation(recurrent_input + external_input)

        # 2) Divisive normalization — §9 PV+ interneuron gain control
        #    r_i = f_drive_i / (σ_inh² + Σ_j W_inh[i,j] · |f_drive_j|)
        #    σ_inh² prevents zero-division and models background inhibitory tone
        if use_divisive_normalization:
            inh_drive = self.W_inh @ np.abs(f_drive)
            res_drive = f_drive / (self.sigma_inh2 + inh_drive)
        else:
            res_drive = f_drive

        # 3) Leaky integrator: dx/dt = -x/τ + r
        dx_dt = -self.x / tau_eff + res_drive

        # 4) Suprathreshold amplification (§10.3): dx/dt += A_amp·x·[S-θ]₊
        if S_target is not None and theta is not None and A_amp > 0:
            margin_plus = max(0.0, S_target - theta)
            dx_dt += A_amp * self.x * margin_plus

        # 5) Euler integration
        self.x = self.x + dt * dx_dt

        return self.x.copy()

    def readout(self, method: str = "linear") -> float:
        """Compute readout signal from reservoir state.

        Spec §10.2: Signal readout

        Args:
            method: Readout method
                - "linear": S = W_out^T·x (trained linear readout)
                - "energy": S = x^T·x (quadratic energy)

        Returns:
            Readout signal (scalar)

        Raises:
            ValueError: If method unknown

        Examples:
            >>> lsm = LiquidStateMachine(N=100, M=2)
            >>> lsm.step(np.array([0.5, -0.3]), tau=1.0)
            >>> S = lsm.readout(method="linear")
            >>> print(type(S))
            <class 'numpy.float64'>
        """
        if method == "linear":
            return float(np.dot(self.W_out.flatten(), self.x))
        elif method == "energy":
            return float(np.dot(self.x, self.x))
        else:
            raise ValueError(f"Unknown readout method: {method}")

    def train_readout(
        self,
        X: np.ndarray,
        y: np.ndarray,
        alpha: float = 1e-6,
    ) -> dict:
        """Train linear readout via ridge regression.

        Spec §10.2: Readout training

        Solves: W_out = argmin_W ||X·W - y||² + α||W||²

        Args:
            X: Reservoir states (T, N)
            y: Target signal (T,)
            alpha: Ridge regularization parameter (default: 1e-6)

        Returns:
            Dictionary with training results:
                - W_out: Trained weights (N, 1)
                - mse: Mean squared error on training data
                - rmse: Root mean squared error
                - r2: R² score

        Raises:
            ValueError: If X, y shapes invalid

        Examples:
            >>> lsm = LiquidStateMachine(N=100, M=2)
            >>> X = np.random.randn(1000, 100)  # Reservoir states
            >>> y = np.random.randn(1000)  # Target signal
            >>> result = lsm.train_readout(X, y, alpha=1e-6)
            >>> print(f"RMSE: {result['rmse']:.4f}")
        """
        X = np.asarray(X)
        y = np.asarray(y)

        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y must have same number of samples: "
                f"X.shape[0]={X.shape[0]}, y.shape[0]={y.shape[0]}"
            )

        if X.shape[1] != self.N:
            raise ValueError(f"X must have {self.N} columns, got {X.shape[1]}")

        # Ridge regression: W_out = (X^T·X + α·I)^{-1}·X^T·y
        gram = X.T @ X + alpha * np.eye(self.N)
        self.W_out = np.linalg.solve(gram, X.T @ y).reshape(-1, 1)

        # Compute training error
        y_pred = X @ self.W_out
        mse = float(np.mean((y_pred - y.reshape(-1, 1)) ** 2))
        rmse = float(np.sqrt(mse))

        # R² score
        ss_res = np.sum((y_pred - y.reshape(-1, 1)) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = float(1.0 - ss_res / (ss_tot + 1e-8))

        return {
            "W_out": self.W_out.copy(),
            "mse": mse,
            "rmse": rmse,
            "r2": r2,
        }

    def collect_state(
        self, state: Optional[np.ndarray] = None, target: Optional[float] = None
    ) -> None:
        """Collect state for training.

        Args:
            state: State to collect (optional, defaults to current self.x)
            target: Target signal value (optional)
        """
        if state is not None:
            self.history.append(state.copy())
        else:
            self.history.append(self.x.copy())
        if target is not None:
            self.history_targets.append(target)

    def get_training_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Get collected states and targets for training.

        Returns:
            (X, y) where X is states (T, N) and y is targets (T,)

        Raises:
            ValueError: If no data collected or no targets collected
        """
        if len(self.history) == 0:
            raise ValueError("No training data collected")

        if len(self.history_targets) == 0:
            raise ValueError("No targets collected")

        X = np.array(self.history)
        y = np.array(self.history_targets)

        return X, y

    def clear_history(self) -> None:
        """Clear collected training data."""
        self.history = []
        self.history_targets = []

    def reset_state(self) -> None:
        """Reset reservoir state to zero."""
        self.x = np.zeros(self.N, dtype=float)

    def _compute_adaptive_tau(
        self,
        precision: float,
        tau_min: float = 0.1,
        tau_max: float = 10.0,
        pi_ref: float = 1.0,
    ) -> float:
        """Compute precision-modulated time constant.

        Higher precision → faster dynamics (lower τ)
        Lower precision → slower dynamics (higher τ)

        Formula: τ(t) = τ_base·(π_ref/π)^α with α=0.5

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

        alpha = 0.5  # Modulation exponent
        tau_adaptive = self.tau_res * (pi_ref / precision) ** alpha

        return float(np.clip(tau_adaptive, tau_min, tau_max))

    def get_state_statistics(self) -> dict:
        """Get statistics of current reservoir state.

        Returns:
            Dictionary with state statistics:
                - mean: Mean of state
                - std: Standard deviation
                - min: Minimum value
                - max: Maximum value
                - norm: L2 norm
        """
        return {
            "mean": float(np.mean(self.x)),
            "std": float(np.std(self.x)),
            "min": float(np.min(self.x)),
            "max": float(np.max(self.x)),
            "norm": float(np.linalg.norm(self.x)),
        }

    def get_weight_statistics(self) -> dict:
        """Get statistics of weight matrices.

        Returns:
            Dictionary with weight statistics
        """
        try:
            with np.errstate(all="ignore"):
                eigs = np.linalg.eigvals(self.W_res)
                spectral_radius = float(np.max(np.abs(eigs)))
        except (np.linalg.LinAlgError, FloatingPointError):
            spectral_radius = 0.0

        inh_sparsity = float(np.mean(self.W_inh > 0))
        return {
            "W_res_spectral_radius": spectral_radius,
            "W_res_mean": float(np.mean(self.W_res)),
            "W_res_std": float(np.std(self.W_res)),
            "W_in_mean": float(np.mean(self.W_in)),
            "W_in_std": float(np.std(self.W_in)),
            "W_out_norm": float(np.linalg.norm(self.W_out)),
            "W_inh_sparsity": inh_sparsity,
            "W_inh_mean_nonzero": (
                float(np.mean(self.W_inh[self.W_inh > 0])) if inh_sparsity > 0 else 0.0
            ),
            "W_inh_norm": float(np.linalg.norm(self.W_inh)),
            "sigma_inh2": self.sigma_inh2,
        }
