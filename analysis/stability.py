"""Fixed-point stability analysis for APGI system.

Implements Jacobian computation and eigenvalue analysis per spec §7.5.

Discrete system at no-ignition fixed point:
    S* = (1-λ)S* + λ·Π|z|  →  S* = Π|z|/(λ)
    θ* = θ* + η[C - V]      →  C = V at equilibrium

Jacobian: J = [[1-λ, 0], [ηc₁λ, e^{-κ}]]

Stability requires: |λ₁| < 1 and |λ₂| < 1
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

# Suppress LAPACK warnings
warnings.filterwarnings("ignore", message=".*On entry to DLASCL.*")


def compute_jacobian_discrete(
    lam: float,
    kappa: float,
    c1: float,
    eta: float,
) -> np.ndarray:
    """Compute Jacobian of discrete APGI system.

    At the no-ignition fixed point, the system is:
        S(t+1) = (1-λ)S(t) + λ·Π|z|
        θ(t+1) = θ(t) + η[C(t) - V(t)] + e^{-κ}(θ(t) - θ_base)

    Linearizing around fixed point:
        J = [[∂S'/∂S, ∂S'/∂θ],
             [∂θ'/∂S, ∂θ'/∂θ]]

    Spec §7.5: J = [[1-λ, 0], [ηc₁λ, e^{-κ}]]

    Args:
        lam: Signal integration rate λ ∈ (0,1)
        kappa: Threshold decay rate κ > 0
        c1: Cost coefficient c₁ > 0
        eta: Threshold learning rate η > 0

    Returns:
        2x2 Jacobian matrix
    """

    J = np.array(
        [
            [1.0 - lam, 0.0],
            [eta * c1 * lam, np.exp(-kappa)],
        ]
    )

    return J


def compute_eigenvalues(J: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute eigenvalues and eigenvectors of Jacobian.

    Args:
        J: Jacobian matrix

    Returns:
        Tuple of (eigenvalues, eigenvectors)
    """
    try:
        with np.errstate(all="ignore"):
            eigs, vecs = np.linalg.eig(J)
            return eigs, vecs
    except (np.linalg.LinAlgError, FloatingPointError):
        # Fallback: return identity-based eigenvalues for 2x2 matrix
        eigs = np.array([J[0, 0], J[1, 1]])
        vecs = np.eye(2)
        return eigs, vecs


def check_stability(
    config: dict,
    verbose: bool = False,
) -> dict[str, Any]:
    """Check fixed-point stability against spec constraints.

    Spec §7.5 Stability Conditions:
    - |λ₁| = 1 - λ < 1  ⟺  λ > 0 ✓ (always satisfied)
    - |λ₂| = e^{-κ} < 1  ⟺  κ > 0 ✓ (always satisfied)

    For stability, both eigenvalues must have magnitude < 1.

    Args:
        config: Configuration dictionary
        verbose: Print detailed analysis

    Returns:
        Dictionary with stability analysis results
    """

    # Extract parameters
    lam = config.get("lam", 0.2)
    kappa = config.get("kappa", 0.15)
    c1 = config.get("c1", 0.2)
    eta = config.get("eta", 0.1)

    # Compute Jacobian
    J = compute_jacobian_discrete(lam, kappa, c1, eta)

    # Compute eigenvalues
    eigs, vecs = compute_eigenvalues(J)

    # Check stability
    eig_magnitudes = np.abs(eigs)
    stable = np.all(eig_magnitudes < 1.0)
    max_eig = np.max(eig_magnitudes)

    # Compute stability margin (how close to instability)
    stability_margin = 1.0 - max_eig

    if verbose:
        print("\n" + "=" * 60)
        print("APGI Fixed-Point Stability Analysis (Spec §7.5)")
        print("=" * 60)
        print("\nParameters:")
        print(f"  λ (integration rate):     {lam:.4f}")
        print(f"  κ (decay rate):           {kappa:.4f}")
        print(f"  c₁ (cost coefficient):    {c1:.4f}")
        print(f"  η (learning rate):        {eta:.4f}")

        print("\nJacobian at no-ignition fixed point:")
        print(f"  J = [[{J[0, 0]:.4f}, {J[0, 1]:.4f}],")
        print(f"       [{J[1, 0]:.4f}, {J[1, 1]:.4f}]]")

        print("\nEigenvalues:")
        for i, eig in enumerate(eigs):
            print(f"  λ_{i + 1} = {eig:.4f} (|λ| = {np.abs(eig):.4f})")

        print("\nStability:")
        print(f"  Max eigenvalue magnitude: {max_eig:.4f}")
        print(f"  Stability margin:         {stability_margin:.4f}")
        print(f"  Status:                   {'STABLE ✓' if stable else 'UNSTABLE ✗'}")

        print("\nConstraint Verification (Spec §7.5):")
        print(f"  λ > 0:                    {lam > 0} ✓")
        print(f"  κ > 0:                    {kappa > 0} ✓")
        print(f"  |λ₁| = 1-λ < 1:           {1 - lam < 1} ✓")
        print(f"  |λ₂| = e^(-κ) < 1:        {np.exp(-kappa) < 1} ✓")
        print("=" * 60 + "\n")

    return {
        "jacobian": J,
        "eigenvalues": eigs,
        "eigenvectors": vecs,
        "eigenvalue_magnitudes": eig_magnitudes,
        "stable": bool(stable),
        "max_eigenvalue": float(max_eig),
        "stability_margin": float(stability_margin),
        "constraints_satisfied": {
            "lambda_positive": bool(lam > 0),
            "kappa_positive": bool(kappa > 0),
            "lambda1_stable": bool(1 - lam < 1),
            "lambda2_stable": bool(np.exp(-kappa) < 1),
        },
    }


def compute_fixed_point(
    config: dict,
) -> dict[str, float]:
    """Compute fixed point of the system.

    At equilibrium (no ignition):
        S* = Π|z| / λ  (from S = (1-λ)S + λ·Π|z|)
        θ* = θ_base    (from θ = θ + η[C - V] with C = V)

    Args:
        config: Configuration dictionary

    Returns:
        Dictionary with fixed point values
    """

    # Signal fixed point depends on precision and error magnitude
    # Use typical values for estimation
    pi_typical = 1.0  # Typical precision
    z_typical = 1.0  # Typical z-score magnitude
    lam = config.get("lam", 0.2)

    S_star = (pi_typical * z_typical) / (lam + 1e-8)

    # Threshold fixed point
    theta_base = config.get("theta_base", 1.0)
    theta_star = theta_base

    return {
        "S_star": float(S_star),
        "theta_star": float(theta_star),
    }


def analyze_bifurcation(
    config: dict,
    param_name: str,
    param_range: tuple[float, float],
    n_points: int = 50,
) -> dict[str, Any]:
    """Analyze bifurcation as parameter varies.

    Computes stability and eigenvalues across parameter range
    to identify bifurcation points.

    Args:
        config: Base configuration
        param_name: Parameter to vary (e.g., "lam", "kappa")
        param_range: (min, max) values for parameter
        n_points: Number of points to sample

    Returns:
        Dictionary with bifurcation analysis
    """

    param_values = np.linspace(param_range[0], param_range[1], n_points)
    eigenvalues_list = []
    stability_list = []

    for param_val in param_values:
        # Create modified config
        cfg = config.copy()
        cfg[param_name] = param_val

        # Compute stability
        result = check_stability(cfg, verbose=False)
        eigenvalues_list.append(result["eigenvalue_magnitudes"])
        stability_list.append(result["stable"])

    eigenvalues_array = np.array(eigenvalues_list)
    stability_array = np.array(stability_list)

    # Find bifurcation points (transitions in stability)
    stability_changes = np.diff(stability_array.astype(int))
    bifurcation_indices = np.where(stability_changes != 0)[0]

    bifurcation_params = []
    if len(bifurcation_indices) > 0:
        for idx in bifurcation_indices:
            bifurcation_params.append(float(param_values[idx]))

    return {
        "parameter_name": param_name,
        "parameter_values": param_values.tolist(),
        "eigenvalue_magnitudes": eigenvalues_array.tolist(),
        "stability": stability_array.tolist(),
        "bifurcation_points": bifurcation_params,
        "stable_region": {
            "min": float(param_values[0]) if stability_array[0] else None,
            "max": float(param_values[-1]) if stability_array[-1] else None,
        },
    }


def validate_system_dynamics(
    config: dict,
    S_history: np.ndarray,
    theta_history: np.ndarray,
) -> dict[str, Any]:
    """Validate system dynamics against theoretical predictions.

    Compares observed dynamics with linearized predictions
    near fixed point.

    Args:
        config: Configuration dictionary
        S_history: Signal history
        theta_history: Threshold history

    Returns:
        Dictionary with validation results
    """

    if len(S_history) < 100:
        return {
            "valid": False,
            "reason": "Insufficient data (need >= 100 samples)",
        }

    # Compute Jacobian
    J = compute_jacobian_discrete(
        config.get("lam", 0.2),
        config.get("kappa", 0.15),
        config.get("c1", 0.2),
        config.get("eta", 0.1),
    )

    # Compute fixed point
    fp = compute_fixed_point(config)
    S_star = fp["S_star"]
    theta_star = fp["theta_star"]

    # Compute deviations from fixed point
    S_dev = S_history - S_star
    theta_dev = theta_history - theta_star

    # Compute observed dynamics
    S_dev_next = S_dev[1:]
    theta_dev_next = theta_dev[1:]
    S_dev_curr = S_dev[:-1]
    theta_dev_curr = theta_dev[:-1]

    # Predict next state using Jacobian
    state_curr = np.array([S_dev_curr, theta_dev_curr])
    state_pred = J @ state_curr

    # Compute prediction error
    state_obs = np.array([S_dev_next, theta_dev_next])
    prediction_error = np.mean((state_pred - state_obs) ** 2)

    # Compute correlation between predicted and observed
    corr_S = np.corrcoef(state_pred[0], state_obs[0])[0, 1]
    corr_theta = np.corrcoef(state_pred[1], state_obs[1])[0, 1]

    return {
        "valid": True,
        "fixed_point": fp,
        "prediction_error": float(prediction_error),
        "correlation_S": float(corr_S),
        "correlation_theta": float(corr_theta),
        "mean_correlation": float((corr_S + corr_theta) / 2),
        "linearization_valid": float((corr_S + corr_theta) / 2) > 0.5,
    }


class StabilityAnalyzer:
    """Comprehensive stability analysis tool for APGI system."""

    def __init__(self, config: dict):
        """Initialize stability analyzer.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.history: dict[str, list[float]] = {
            "S": [],
            "theta": [],
        }

    def step(self, S: float, theta: float) -> None:
        """Record state for analysis.

        Args:
            S: Signal value
            theta: Threshold value
        """
        self.history["S"].append(S)
        self.history["theta"].append(theta)

    def analyze(self, verbose: bool = False) -> dict[str, Any]:
        """Perform comprehensive stability analysis.

        Args:
            verbose: Print detailed output

        Returns:
            Dictionary with complete analysis
        """

        # Fixed-point stability
        stability = check_stability(self.config, verbose=verbose)

        # Fixed point
        fixed_point = compute_fixed_point(self.config)

        # Dynamics validation (if enough history)
        if len(self.history["S"]) >= 100:
            dynamics = validate_system_dynamics(
                self.config,
                np.array(self.history["S"]),
                np.array(self.history["theta"]),
            )
        else:
            dynamics = {
                "valid": False,
                "reason": "Insufficient history",
            }

        return {
            "stability": stability,
            "fixed_point": fixed_point,
            "dynamics_validation": dynamics,
        }
