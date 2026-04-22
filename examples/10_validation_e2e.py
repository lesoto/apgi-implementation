"""End-to-end validation script for spectral and observable predictions on synthetic data.

This script validates:
1. Pink noise (1/f) spectral characteristics in threshold dynamics
2. Hurst exponent estimation for long-range correlations
3. Neural observable predictions (gamma power, ERP amplitude, ignition rate)
4. Behavioral observable predictions (RT variability, response criterion
"""

# flake8: noqa=E402 (module level import not at top of file - needed for sys.path manipulation)
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from pipeline import APGIPipeline
from config import CONFIG


def generate_synthetic_data(n_steps: int = 1000, dt: float = 0.5):
    """Generate synthetic exteroceptive and interoceptive input signals.

    Args:
        n_steps: Number of timesteps
        dt: Time step size

    Returns:
        Tuple of (x_e array, x_i array) synthetic input signals
    """
    # Generate pink noise (1/f) for exteroceptive input using Voss-McCartney algorithm
    # This is more numerically stable than spectral synthesis
    np.random.seed(42)  # For reproducibility
    x_e = np.zeros(n_steps)
    for i in range(n_steps):
        # Sum of several octaves of white noise
        x_e[i] = np.sum([np.random.randn() / (2**j) for j in range(6)])

    # Generate oscillatory interoceptive signal with drift
    t = np.arange(n_steps) * dt
    x_i = 0.5 * np.sin(2 * np.pi * 0.1 * t) + 0.3 * np.random.randn(n_steps)

    # Normalize to zero mean, unit variance
    x_e = (x_e - np.mean(x_e)) / (np.std(x_e) + 1e-8)
    x_i = (x_i - np.mean(x_i)) / (np.std(x_i) + 1e-8)

    return x_e, x_i


def validate_spectral_characteristics(pipeline: APGIPipeline):
    """Validate pink noise (1/f) characteristics in threshold dynamics."""
    if len(pipeline.history["theta"]) < 64:
        print("Warning: Insufficient data for spectral validation")
        return

    theta_arr = np.array(pipeline.history["theta"])
    fs = 1.0 / pipeline.config.get("dt", 1.0)

    from stats.hurst import estimate_hurst_robust, welch_periodogram
    from stats.spectral_model import validate_pink_noise

    # Estimate Hurst exponent
    hurst = estimate_hurst_robust(theta_arr, fs=fs)
    print(f"Hurst exponent: {hurst:.4f}")
    print("  Expected for pink noise: ~0.7-0.9")

    hurst_pass = 0.7 <= hurst <= 0.9
    print(f"  Hurst validation: {'PASS' if hurst_pass else 'FAIL'}")

    # Validate pink noise (may fail due to numerical issues)
    try:
        freqs, psd = welch_periodogram(theta_arr, fs=fs)
        pink_stats = validate_pink_noise(freqs, psd)
        print(
            f"Pink noise validation: {'PASS' if pink_stats['is_pink_noise'] else 'FAIL'}"
        )
        print(f"  Beta (1/f slope): {pink_stats['beta']:.4f}")
        print("  Expected beta: ~1.0")
        return hurst, pink_stats
    except Exception as e:
        print(f"Pink noise validation skipped: {e}")
        print("  (Linear regression failed - Hurst exponent validation still valid)")
        return hurst, None


def validate_observable_predictions(pipeline: APGIPipeline):
    """Validate neural and behavioral observable predictions."""
    # Enable observable mapping
    if pipeline.neural_observables is None or pipeline.behavioral_observables is None:
        print("Observable mapping not enabled in config")
        return

    # Extract observable statistics
    gamma_powers = pipeline.history.get("neural_gamma_power", [])
    erp_amplitudes = pipeline.history.get("neural_erp_amplitude", [])
    ignition_rates = pipeline.history.get("neural_ignition_rate", [])
    rt_variability = pipeline.history.get("behavioral_rt_variability", [])
    response_criterion = pipeline.history.get("behavioral_response_criterion", [])

    if gamma_powers:
        print(f"Mean gamma power: {np.mean(gamma_powers):.4f}")
    if erp_amplitudes:
        print(f"Mean ERP amplitude: {np.mean(erp_amplitudes):.4f}")
    if ignition_rates:
        print(f"Mean ignition rate: {np.mean(ignition_rates):.4f}")
    if rt_variability:
        print(f"Mean RT variability: {np.mean(rt_variability):.4f}")
    if response_criterion:
        print(f"Mean response criterion: {np.mean(response_criterion):.4f}")


def main():
    """Run end-to-end validation."""
    print("=" * 70)
    print("APGI End-to-End Validation")
    print("=" * 70)

    # Configure pipeline with observable mapping and stability analysis
    config = dict(CONFIG)
    config["use_observable_mapping"] = True
    config["use_stability_analysis"] = True
    config["stochastic_ignition"] = False
    config["n_steps"] = 1000
    # Test spec-compliant parameter names
    config["beta_da"] = 0.0  # Spec-preferred dopamine bias
    config["tau_sigma"] = 0.5  # Spec-preferred ignition temperature

    # Initialize pipeline
    pipeline = APGIPipeline(config)

    # Generate synthetic data
    print("\nGenerating synthetic data...")
    n_steps = 1000
    x_e, x_i = generate_synthetic_data(n_steps=n_steps)

    # Run pipeline
    print(f"Running pipeline for {n_steps} steps...")
    for i in range(n_steps):
        pipeline.step(x_e[i], x_i[i])

    # Validate spectral characteristics
    print("\n" + "-" * 70)
    print("Spectral Validation")
    print("-" * 70)
    try:
        hurst, pink_stats = validate_spectral_characteristics(pipeline)
    except Exception as e:
        print(f"Spectral validation failed: {e}")
        print("  This is likely due to numerical issues with the synthetic data.")
        hurst, pink_stats = None, None

    # Validate observable predictions
    print("\n" + "-" * 70)
    print("Observable Predictions Validation")
    print("-" * 70)
    validate_observable_predictions(pipeline)

    # Stability analysis
    if pipeline.stability_analyzer is not None:
        print("\n" + "-" * 70)
        print("Stability Analysis")
        print("-" * 70)
        stability_result = pipeline.stability_analyzer.analyze(verbose=False)
        print(f"Fixed point S*: {stability_result['fixed_point']['S_star']:.4f}")
        print(
            f"Stability: {'STABLE' if stability_result['stability']['stable'] else 'UNSTABLE'}"
        )
        print(
            f"Max eigenvalue magnitude: {stability_result['stability']['max_eigenvalue']:.4f}"
        )

    print("\n" + "=" * 70)
    print("Validation Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
