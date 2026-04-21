"""APGI (Adaptive Precision Gated Ignition) - Main Entry Point

Example usage:
    python main.py --steps 1000 --output results.json
    python main.py --demo  # Run demonstration with synthetic data
    python main.py --multiscale --levels 5  # Enable multi-scale hierarchy
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import numpy as np

from config import CONFIG
from core.precision import compute_precision
from hierarchy.multiscale import (
    aggregate_multiscale_signal,
    build_timescales,
    multiscale_weights,
    update_multiscale_feature,
)
from pipeline import APGIPipeline
from stats.hurst import estimate_hurst_robust


def generate_synthetic_input(
    t: int, noise_std: float = 0.1
) -> tuple[float, float, float, float]:
    """Generate synthetic exteroceptive and interoceptive signals.

    Returns:
        (x_e, x_hat_e, x_i, x_hat_i) - actual and predicted values
    """

    # Exteroceptive: external sensory signal with periodic component
    x_e = np.sin(0.05 * t) + 0.5 * np.sin(0.01 * t) + np.random.normal(0, noise_std)
    x_hat_e = np.sin(0.05 * t)  # Predictor sees only fast component

    # Interoceptive: internal signal (e.g., heart rate variability)
    x_i = 0.5 + 0.3 * np.cos(0.02 * t) + np.random.normal(0, noise_std * 0.5)
    x_hat_i = 0.5  # Baseline prediction

    return float(x_e), float(x_hat_e), float(x_i), float(x_hat_i)


def run_standard_pipeline(
    n_steps: int = 1000,
    config: dict[str, Any] | None = None,
    progress_interval: int = 100,
) -> dict[str, Any]:
    """Run standard single-scale APGI pipeline.

    Args:
        n_steps: Number of simulation steps
        config: Optional custom configuration (uses default CONFIG if None)
        progress_interval: Print progress every N steps

    Returns:
        Dictionary with simulation results and statistics
    """

    cfg = config or CONFIG.copy()
    pipeline = APGIPipeline(cfg)

    history: dict[str, list[float]] = {
        "S": [],
        "theta": [],
        "B": [],
        "z_e": [],
        "z_i": [],
        "p_ignite": [],
        "C": [],
        "V": [],
    }

    print(f"Running standard APGI pipeline for {n_steps} steps...")
    print(f"Initial threshold: {pipeline.theta:.4f}, Initial signal: {pipeline.S:.4f}")

    ignition_count = 0

    for t in range(n_steps):
        x_e, x_hat_e, x_i, x_hat_i = generate_synthetic_input(t)
        result = pipeline.step(x_e, x_hat_e, x_i, x_hat_i)

        for key in history:
            history[key].append(result[key])

        if result["B"] == 1:
            ignition_count += 1

        if (t + 1) % progress_interval == 0:
            print(
                f"  Step {t + 1}/{n_steps}: S={result['S']:.4f}, θ={result['theta']:.4f}, "
                f"P(ignite)={result['p_ignite']:.4f}, B={result['B']}"
            )

    print(
        f"\nCompleted: {ignition_count} ignition events ({100 * ignition_count / n_steps:.1f}%)"
    )
    print(f"Final signal: {pipeline.S:.4f}, Final threshold: {pipeline.theta:.4f}")

    return {
        "config": cfg,
        "n_steps": n_steps,
        "ignition_count": ignition_count,
        "ignition_rate": ignition_count / n_steps,
        "history": history,
        "final_state": {
            "S": pipeline.S,
            "theta": pipeline.theta,
            "sigma2_e": pipeline.state.sigma2_e,
            "sigma2_i": pipeline.state.sigma2_i,
        },
    }


def run_multiscale_pipeline(
    n_steps: int = 1000,
    n_levels: int = 5,
    timescale_k: float = 1.6,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run multi-scale APGI pipeline (hierarchical timescales).

    Args:
        n_steps: Number of simulation steps
        n_levels: Number of timescale hierarchy levels
        timescale_k: Timescale expansion factor (recommended: 1.3-2.0)
        config: Optional custom configuration

    Returns:
        Dictionary with simulation results
    """

    cfg = config or CONFIG.copy()
    cfg["timescale_k"] = timescale_k

    tau0 = 1.0
    taus = build_timescales(tau0, timescale_k, n_levels)
    weights = multiscale_weights(n_levels, timescale_k)

    print(f"Running multi-scale APGI pipeline ({n_levels} levels, k={timescale_k})...")
    print(f"Timescales: {taus}")
    print(f"Weights: {weights}")

    # Initialize multi-scale feature buffers and per-level EMA variances
    phi_e = np.zeros(n_levels)
    phi_i = np.zeros(n_levels)
    sigma2_levels = np.ones(n_levels)  # per-level combined variance estimate

    pipeline = APGIPipeline(cfg)
    history: dict[str, list[float]] = {
        "S_multiscale": [],
        "S_standard": [],
        "B": [],
        "theta": [],
    }
    ignition_count = 0

    for t in range(n_steps):
        x_e, x_hat_e, x_i, x_hat_i = generate_synthetic_input(t)
        z_e = x_e - x_hat_e
        z_i = x_i - x_hat_i

        # Update multi-scale features and per-level variance estimates
        for i in range(n_levels):
            phi_e[i] = update_multiscale_feature(phi_e[i], z_e, taus[i])
            phi_i[i] = update_multiscale_feature(phi_i[i], z_i, taus[i])
            # EMA rate proportional to 1/τ_i; min to keep it in (0,1]
            alpha_l = min(1.0, 1.0 / taus[i])
            sigma2_levels[i] = (1.0 - alpha_l) * sigma2_levels[i] + alpha_l * (
                phi_e[i] ** 2 + phi_i[i] ** 2
            )

        # Per-level precision from actual variance estimates
        pi_levels = [
            compute_precision(
                sigma2_levels[i], cfg["eps"], cfg["pi_min"], cfg["pi_max"]
            )
            for i in range(n_levels)
        ]
        S_multiscale = aggregate_multiscale_signal(phi_e + phi_i, pi_levels, weights)

        # Run standard pipeline for comparison
        result = pipeline.step(x_e, x_hat_e, x_i, x_hat_i)

        history["S_multiscale"].append(S_multiscale)
        history["S_standard"].append(result["S"])
        history["B"].append(result["B"])
        history["theta"].append(result["theta"])

        if result["B"] == 1:
            ignition_count += 1

    print(f"\nCompleted: {ignition_count} ignition events")
    print(
        f"Multi-scale S range: [{min(history['S_multiscale']):.4f}, {max(history['S_multiscale']):.4f}]"
    )
    print(
        f"Standard S range: [{min(history['S_standard']):.4f}, {max(history['S_standard']):.4f}]"
    )

    return {
        "config": cfg,
        "n_steps": n_steps,
        "n_levels": n_levels,
        "timescales": taus.tolist(),
        "weights": weights.tolist(),
        "ignition_count": ignition_count,
        "history": history,
    }


def analyze_signal_statistics(
    signal_history: list[float], label: str = "Signal"
) -> dict[str, float]:
    """Compute statistics and estimate Hurst exponent from signal history.

    Args:
        signal_history: Time series of signal values
        label: Label for print output

    Returns:
        Dictionary with computed statistics
    """

    signal = np.array(signal_history)

    stats = {
        "mean": float(np.mean(signal)),
        "std": float(np.std(signal)),
        "min": float(np.min(signal)),
        "max": float(np.max(signal)),
        "range": float(np.max(signal) - np.min(signal)),
    }

    # Estimate Hurst exponent if enough data
    if len(signal) >= 256:
        try:
            H = estimate_hurst_robust(signal, fs=1.0, method="welch")
            stats["hurst_exponent"] = H
            print("\n" + label + " Statistics:")
            print("  Mean: %.4f, Std: %.4f" % (stats["mean"], stats["std"]))
            print("  Range: [%.4f, %.4f]" % (stats["min"], stats["max"]))
            print("  Hurst Exponent (H): %.4f" % H)
            if H > 0.5:
                print("  → Persistent/long-memory process (H > 0.5)")
            elif H < 0.5:
                print("  → Anti-persistent process (H < 0.5)")
            else:
                print("  → Random walk (H ≈ 0.5)")
        except Exception as e:
            print("  Hurst estimation failed: " + str(e))
    else:
        print("\n" + label + " Statistics (insufficient data for Hurst):")
        for key, value in stats.items():
            print("  " + key + ": %.4f" % value)

    return stats


def save_results(results: dict[str, Any], filepath: str) -> None:
    """Save simulation results to JSON file.

    Args:
        results: Dictionary with simulation results
        filepath: Output file path
    """

    # Convert numpy arrays to lists for JSON serialization
    def convert(obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [convert(v) for v in obj]
        return obj

    serializable = convert(results)

    with open(filepath, "w") as f:
        json.dump(serializable, f, indent=2)

    print(f"\nResults saved to: {filepath}")


def main() -> int:
    """Main entry point."""

    parser = argparse.ArgumentParser(
        description="APGI (Adaptive Precision Gated Ignition) Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --demo                    # Quick demonstration
  %(prog)s --steps 5000 --output out.json   # Long run with save
  %(prog)s --multiscale --levels 7   # Multi-scale hierarchy
  %(prog)s --ne-on-threshold         # Use NE on threshold (not precision)
        """,
    )

    parser.add_argument("--demo", action="store_true", help="Run quick demonstration")
    parser.add_argument(
        "--steps", type=int, default=1000, help="Number of simulation steps"
    )
    parser.add_argument("--output", type=str, help="Save results to JSON file")
    parser.add_argument(
        "--multiscale", action="store_true", help="Enable multi-scale hierarchy"
    )
    parser.add_argument(
        "--levels",
        type=int,
        default=5,
        help="Number of hierarchy levels (with --multiscale)",
    )
    parser.add_argument(
        "--k", type=float, default=1.6, help="Timescale expansion factor (1.3-2.0)"
    )
    parser.add_argument(
        "--ne-on-threshold",
        action="store_true",
        help="Apply NE to threshold instead of precision",
    )
    parser.add_argument(
        "--beta", type=float, default=0.0, help="Dopaminergic bias value"
    )
    parser.add_argument(
        "--stochastic", action="store_true", help="Enable stochastic ignition"
    )
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")

    args = parser.parse_args()

    # Set random seed if provided
    if args.seed is not None:
        np.random.seed(args.seed)

    # Build custom config from defaults + CLI overrides
    config = CONFIG.copy()
    config["beta"] = args.beta
    config["stochastic_ignition"] = args.stochastic

    if args.ne_on_threshold:
        config["ne_on_precision"] = False
        config["ne_on_threshold"] = True
        print("Configuration: NE modulates threshold (not precision)")

    print("\n" + "=" * 60)
    print("APGI - Adaptive Precision Gated Ignition")
    print("=" * 60)

    try:
        if args.multiscale:
            results = run_multiscale_pipeline(
                n_steps=args.steps,
                n_levels=args.levels,
                timescale_k=args.k,
                config=config,
            )
            analyze_signal_statistics(
                results["history"]["S_multiscale"], "Multi-scale Signal"
            )
        else:
            results = run_standard_pipeline(n_steps=args.steps, config=config)
            analyze_signal_statistics(results["history"]["S"], "Standard Signal")

        if args.output:
            save_results(results, args.output)

        print("\n" + "=" * 60)
        print("Simulation complete.")
        print("=" * 60)

        return 0

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        return 130
    except Exception as e:
        print(f"\nError: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
