"""Hierarchical multi-timescale APGI system example.

This example demonstrates the full hierarchical system with:
- Multi-scale error processing (per-level z-scores)
- Precision coupling ODE (top-down and bottom-up)
- Phase-amplitude coupling (PAC)
- Bottom-up threshold cascade

The hierarchical system implements multi-scale processing where each level
operates at its own timescale, producing 1/f spectral characteristics.
"""

# flake8: noqa=E402 (module level import not at top of file - needed for sys.path manipulation)
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import numpy.typing as npt
from typing import cast

from config import CONFIG
from hierarchy.multiscale import build_timescales, multiscale_weights
from pipeline import APGIPipeline


def generate_input_signals(
    n_steps: int, dt: float = 0.1
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Generate synthetic exteroceptive and interoceptive input signals.

    Args:
        n_steps: Number of timesteps
        dt: Time step size

    Returns:
        Tuple of (x_e array, x_i array) input signals
    """
    rng = np.random.default_rng(42)

    # Exteroceptive: multi-frequency signal (simulating visual/audio input)
    t = np.arange(n_steps) * dt
    x_e = (
        np.sin(2 * np.pi * 0.5 * t)  # Slow component
        + 0.5 * np.sin(2 * np.pi * 2.0 * t)  # Fast component
        + 0.3 * rng.standard_normal(n_steps)  # Noise
    )

    # Interoceptive: heart-rate-like oscillatory signal
    x_i = (
        np.sin(2 * np.pi * 1.2 * t)
        + 0.3 * np.sin(2 * np.pi * 0.3 * t)  # Slower modulation
        + 0.2 * rng.standard_normal(n_steps)
    )

    return x_e, x_i


def run_basic_hierarchical() -> None:
    """Run basic hierarchical mode example."""
    print("\n" + "=" * 70)
    print("Example 1: Basic Hierarchical Mode")
    print("=" * 70)

    config = dict(CONFIG)
    config["hierarchical_mode"] = "basic"
    config["n_levels"] = 3
    config["tau_0"] = 10.0
    config["k"] = 1.6
    config["stochastic_ignition"] = False

    print(f"\nConfiguration:")
    print(f"  Mode: basic")
    print(f"  Levels: {config['n_levels']}")
    print(f"  Base timescale τ_0: {config['tau_0']} ms")
    print(f"  Timescale ratio k: {config['k']}")

    pipeline = APGIPipeline(config)

    # Show timescales
    tau_0 = cast(float, config["tau_0"])
    k = cast(float, config["k"])
    n_levels = cast(int, config["n_levels"])
    taus = build_timescales(tau_0, k, n_levels)
    print(f"\nComputed timescales: {taus} ms")

    # Generate and process data
    n_steps = 500
    x_e, x_i = generate_input_signals(n_steps)

    print(f"\nProcessing {n_steps} steps...")

    results = []
    for i in range(n_steps):
        result = pipeline.step(x_e[i], x_i[i])
        results.append(result)

    # Analyze results
    ignition_count = sum(1 for r in results if r["B"] == 1)
    print(f"\nResults:")
    print(f"  Ignition events: {ignition_count} ({100*ignition_count/n_steps:.1f}%)")
    print(f"  Mean signal S: {np.mean([r['S'] for r in results]):.4f}")
    print(f"  Mean threshold θ: {np.mean([r['theta'] for r in results]):.4f}")

    if "hierarchical_pis" in results[0]:
        final_pis = results[-1]["hierarchical_pis"]
        print(f"  Final per-level precisions: {final_pis}")


def run_advanced_hierarchical() -> None:
    """Run advanced hierarchical mode with precision ODE."""
    print("\n" + "=" * 70)
    print("Example 2: Advanced Hierarchical Mode (with Precision ODE)")
    print("=" * 70)

    config = dict(CONFIG)
    config["hierarchical_mode"] = "advanced"
    config["n_levels"] = 4
    config["tau_0"] = 10.0
    config["k"] = 1.6

    # Precision ODE parameters
    config["tau_pi"] = 1000.0  # Precision timescale (ms)
    config["C_down"] = 0.1  # Top-down coupling
    config["C_up"] = 0.05  # Bottom-up coupling
    config["alpha_gain"] = 0.1  # Error-to-precision gain

    print(f"\nConfiguration:")
    print(f"  Mode: advanced")
    print(f"  Levels: {config['n_levels']}")
    print(f"  Precision ODE: enabled")
    print(f"    τ_π: {config['tau_pi']} ms")
    print(f"    C_down: {config['C_down']}")
    print(f"    C_up: {config['C_up']}")

    pipeline = APGIPipeline(config)

    n_steps = 500
    x_e, x_i = generate_input_signals(n_steps)

    print(f"\nProcessing {n_steps} steps...")

    # Track precision evolution
    precision_history = []

    for i in range(n_steps):
        result = pipeline.step(x_e[i], x_i[i])
        if "hierarchical_pis" in result:
            precision_history.append(result["hierarchical_pis"])

    print(f"\nResults:")
    if precision_history:
        initial_pis = precision_history[0]
        final_pis = precision_history[-1]
        print(f"  Initial precisions: {[f'{p:.2f}' for p in initial_pis]}")
        print(f"  Final precisions: {[f'{p:.2f}' for p in final_pis]}")

        # Show precision evolution
        print(f"\nPrecision evolution (last 5 steps):")
        for i, pis in enumerate(precision_history[-5:]):
            print(f"    Step {-5+i}: {[f'{p:.2f}' for p in pis]}")


def run_full_hierarchical() -> None:
    """Run full hierarchical mode with all features."""
    print("\n" + "=" * 70)
    print("Example 3: Full Hierarchical Mode (all features)")
    print("=" * 70)

    config = dict(CONFIG)
    config["hierarchical_mode"] = "full"
    config["n_levels"] = 5
    config["tau_0"] = 10.0
    config["k"] = 1.6

    # Precision ODE parameters
    config["tau_pi"] = 1000.0
    config["C_down"] = 0.1
    config["C_up"] = 0.05
    config["alpha_gain"] = 0.1

    # Phase modulation parameters
    config["kappa_phase"] = 0.1
    config["omega_phases"] = [0.1, 0.05, 0.025, 0.0125, 0.00625]

    print(f"\nConfiguration:")
    print(f"  Mode: full")
    print(f"  Levels: {config['n_levels']}")
    print(f"  Precision ODE: enabled")
    print(f"  Phase modulation: enabled")
    print(f"    κ_phase: {config['kappa_phase']}")
    print(f"    ω_phases: {config['omega_phases']}")

    pipeline = APGIPipeline(config)

    n_steps = 500
    x_e, x_i = generate_input_signals(n_steps)

    print(f"\nProcessing {n_steps} steps...")

    # Track all hierarchical state
    history: dict[str, list] = {
        "pis": [],
        "phases": [],
        "thetas": [],
        "ignitions": [],
    }

    for i in range(n_steps):
        result = pipeline.step(x_e[i], x_i[i])

        if "hierarchical_pis" in result:
            history["pis"].append(result["hierarchical_pis"])
        if "hierarchical_phases" in result:
            history["phases"].append(result["hierarchical_phases"])
        if "hierarchical_thetas" in result:
            history["thetas"].append(result["hierarchical_thetas"])
        history["ignitions"].append(result["B"])

    print(f"\nResults:")
    ignition_count = sum(history["ignitions"])
    print(f"  Ignition events: {ignition_count}")

    if history["pis"]:
        print(f"\n  Final hierarchical state:")
        print(f"    Precisions (Π): {[f'{p:.2f}' for p in history['pis'][-1]]}")
        print(f"    Phases (φ): {[f'{p:.3f}' for p in history['phases'][-1]]}")

    # Demonstrate per-level error computation
    print(f"\n  Per-level error processing:")
    print(f"    Each level computes z-scores at its own timescale")
    tau_0_full = cast(float, config["tau_0"])
    k_full = cast(float, config["k"])
    n_levels_full = cast(int, config["n_levels"])
    print(f"    Level 0 (τ={tau_0_full}ms): Fast adaptation")
    print(
        f"    Level {n_levels_full-1} (τ={tau_0_full*(k_full**(n_levels_full-1)):.1f}ms): Slow adaptation"
    )


def compare_hierarchical_modes() -> None:
    """Compare different hierarchical modes side by side."""
    print("\n" + "=" * 70)
    print("Example 4: Comparing Hierarchical Modes")
    print("=" * 70)

    n_steps = 300
    x_e, x_i = generate_input_signals(n_steps)

    modes = ["off", "basic", "advanced", "full"]
    results_by_mode = {}

    for mode in modes:
        config = dict(CONFIG)
        config["hierarchical_mode"] = mode
        config["n_levels"] = 3
        config["tau_0"] = 10.0
        config["k"] = 1.6
        config["stochastic_ignition"] = False

        pipeline = APGIPipeline(config)

        signals = []
        thresholds = []
        ignitions = []

        for i in range(n_steps):
            result = pipeline.step(x_e[i], x_i[i])
            signals.append(result["S"])
            thresholds.append(result["theta"])
            ignitions.append(result["B"])

        results_by_mode[mode] = {
            "signal_var": np.var(signals),
            "threshold_var": np.var(thresholds),
            "ignition_rate": np.mean(ignitions),
        }

    print(f"\nComparison (n={n_steps} steps):")
    print(f"{'Mode':<12} {'Signal Var':>12} {'Threshold Var':>14} {'Ignition Rate':>14}")
    print("-" * 60)
    for mode in modes:
        r = results_by_mode[mode]
        print(
            f"{mode:<12} {r['signal_var']:>12.4f} {r['threshold_var']:>14.4f} {r['ignition_rate']:>14.4f}"
        )

    print(f"\nObservations:")
    print(f"  - 'off': Single-scale processing (baseline)")
    print(f"  - 'basic': Multi-scale integration without coupling")
    print(f"  - 'advanced': Adds precision ODE dynamics")
    print(f"  - 'full': Adds phase-amplitude coupling")


def demonstrate_timescales() -> None:
    """Demonstrate how timescales are computed."""
    print("\n" + "=" * 70)
    print("Example 5: Understanding Timescale Hierarchy")
    print("=" * 70)

    tau_0 = 10.0  # Base timescale (ms)
    k_values = [1.3, 1.6, 2.0]  # Different ratios
    n_levels = 5

    print(f"\nBase timescale τ_0 = {tau_0} ms")
    print(f"Formula: τ_ℓ = τ_0 · k^ℓ")

    for k in k_values:
        taus = build_timescales(tau_0, k, n_levels)
        weights = multiscale_weights(n_levels, k)

        print(f"\n  k = {k} (ratio):")
        for i, (tau, w) in enumerate(zip(taus, weights)):
            print(f"    Level {i}: τ = {tau:6.2f} ms, weight = {w:.4f}")

    print(f"\nNote: Weights decrease geometrically with level")
    print(f"      Higher levels (slower) contribute less to immediate signal")


def main() -> None:
    """Run all hierarchical system examples."""
    print("=" * 70)
    print("APGI Hierarchical Multi-Timescale System Examples")
    print("=" * 70)

    # Run all examples
    demonstrate_timescales()
    run_basic_hierarchical()
    run_advanced_hierarchical()
    run_full_hierarchical()
    compare_hierarchical_modes()

    print("\n" + "=" * 70)
    print("All Hierarchical Examples Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
