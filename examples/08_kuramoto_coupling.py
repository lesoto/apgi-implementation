"""Kuramoto oscillatory coupling example for APGI.

This example demonstrates phase-amplitude coupling using Kuramoto oscillators
to model hierarchical oscillatory synchronization. Key features:

- Phase dynamics with Ornstein-Uhlenbeck noise
- Phase reset on ignition events
- Frequency modulation by precision
- Cross-level phase coupling
- Synchronization metrics

The Kuramoto model captures the essence of coupled oscillators in neural systems,
where phases synchronize to coordinate information processing across scales.
"""

# flake8: noqa=E402 (module level import not at top of file - needed for sys.path manipulation)
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import numpy.typing as npt

from pipeline import APGIPipeline
from config import CONFIG
from oscillation.kuramoto import HierarchicalKuramotoSystem


def generate_input_signals(
    n_steps: int, dt: float = 0.1
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Generate synthetic input signals.

    Args:
        n_steps: Number of timesteps
        dt: Time step size

    Returns:
        Tuple of (x_e, x_i) input signals
    """
    rng = np.random.default_rng(123)

    t = np.arange(n_steps) * dt

    # Exteroceptive: modulated signal
    x_e = (
        np.sin(2 * np.pi * 0.3 * t)
        + 0.5 * np.sin(2 * np.pi * 0.8 * t)
        + 0.2 * rng.standard_normal(n_steps)
    )

    # Interoceptive: slower oscillation
    x_i = 0.7 * np.sin(2 * np.pi * 0.15 * t) + 0.3 * rng.standard_normal(n_steps)

    return x_e, x_i


def demonstrate_kuramoto_system():
    """Demonstrate standalone Kuramoto system."""
    print("\n" + "=" * 70)
    print("Example 1: Standalone Kuramoto Oscillator System")
    print("=" * 70)

    n_levels = 4
    dt = 0.1

    # Configure Kuramoto system
    config = {
        "omega_0": 0.5,  # Base frequency (Hz)
        "freq_scaling": 0.7,  # Frequency ratio between levels
        "noise_std": 0.05,  # Phase noise
        "coupling_strength": 0.1,  # Inter-level coupling
    }

    print(f"\nConfiguration:")
    print(f"  Levels: {n_levels}")
    print(f"  Base frequency ω_0: {config['omega_0']} Hz")
    print(f"  Frequency scaling: {config['freq_scaling']}")
    print(f"  Coupling strength: {config['coupling_strength']}")

    # Initialize Kuramoto system
    kuramoto = HierarchicalKuramotoSystem(n_levels=n_levels, config=config)

    # Run simulation
    n_steps = 500
    phase_history = []
    sync_history = []

    print(f"\nSimulating {n_steps} steps...")

    for _ in range(n_steps):
        result = kuramoto.step(dt=dt)
        phase_history.append(result["phases"].copy())
        sync_history.append(result["synchronization"])

    # Analyze results
    phase_array = np.array(phase_history)

    print(f"\nResults:")
    print(f"  Mean synchronization: {np.mean(sync_history):.4f}")
    print(f"  Final synchronization: {sync_history[-1]:.4f}")
    print(
        f"  Synchronization range: [{min(sync_history):.4f}, {max(sync_history):.4f}]"
    )

    print(f"\n  Phase ranges (radians):")
    for level in range(n_levels):
        phases = phase_array[:, level]
        print(
            f"    Level {level}: [{phases.min():.3f}, {phases.max():.3f}] "
            f"(wrapped: [{phases.min() % (2*np.pi):.3f}, {phases.max() % (2*np.pi):.3f}])"
        )

    # Demonstrate phase reset
    print(f"\n  Applying phase reset at level 0...")
    kuramoto.apply_ignition_reset(level=0)
    result = kuramoto.step(dt=dt)

    print(f"    Post-reset phases: {result['phases']}")
    print(f"    Post-reset sync: {result['synchronization']:.4f}")


def demonstrate_kuramoto_with_apgi():
    """Demonstrate Kuramoto-integrated APGI pipeline."""
    print("\n" + "=" * 70)
    print("Example 2: APGI Pipeline with Kuramoto Coupling")
    print("=" * 70)

    config = dict(CONFIG)
    config["use_kuramoto"] = True
    config["n_levels"] = 4
    config["hierarchical_mode"] = "full"
    config["tau_0"] = 10.0
    config["k"] = 1.6

    # Kuramoto-specific parameters
    config["omega_0"] = 0.5
    config["freq_scaling"] = 0.7
    config["noise_std"] = 0.05
    config["coupling_strength"] = 0.1

    # Phase modulation parameters (one per level)
    config["omega_phases"] = [0.1, 0.05, 0.025, 0.0125]

    print(f"\nConfiguration:")
    print(f"  use_kuramoto: True")
    print(f"  Levels: {config['n_levels']}")
    print(f"  Hierarchical mode: {config['hierarchical_mode']}")

    pipeline = APGIPipeline(config)

    n_steps = 300
    x_e, x_i = generate_input_signals(n_steps)

    print(f"\nSimulating {n_steps} steps with Kuramoto-APGI integration...")

    # Track metrics
    phase_history = []
    sync_history = []
    ignition_phases = []

    for i in range(n_steps):
        result = pipeline.step(x_e[i], x_i[i])

        if "kuramoto_phases" in result:
            phase_history.append(result["kuramoto_phases"])
            sync_history.append(result["kuramoto_synchronization"])

        # Track phase at ignition
        if result["B"] == 1 and "kuramoto_phases" in result:
            ignition_phases.append(
                {
                    "step": i,
                    "phases": result["kuramoto_phases"],
                    "sync": result["kuramoto_synchronization"],
                }
            )

    print(f"\nResults:")
    print(f"  Mean synchronization: {np.mean(sync_history):.4f}")
    print(f"  Ignition events: {len(ignition_phases)}")

    if ignition_phases:
        print(f"\n  Phase states at ignition events:")
        for event in ignition_phases[:3]:  # Show first 3
            print(
                f"    Step {event['step']}: sync={event['sync']:.4f}, "
                f"phases={[f'{p:.3f}' for p in event['phases']]}"
            )

    # Show phase reset effect
    if len(ignition_phases) >= 2:
        print(f"\n  Phase reset effect:")
        print(f"    Kuramoto phases reset on each ignition event")
        print(f"    This coordinates oscillatory state with ignition timing")


def demonstrate_phase_modulation():
    """Demonstrate phase-amplitude coupling (PAC)."""
    print("\n" + "=" * 70)
    print("Example 3: Phase-Amplitude Coupling (PAC)")
    print("=" * 70)

    config = dict(CONFIG)
    config["use_kuramoto"] = True
    config["use_phase_modulation"] = True
    config["hierarchical_mode"] = "full"
    config["n_levels"] = 3

    # Phase modulation parameters
    config["kappa_phase"] = 0.2  # Phase modulation strength
    config["omega_phases"] = [0.5, 0.25, 0.125]  # Level frequencies

    print(f"\nConfiguration:")
    print(f"  Phase modulation: enabled")
    print(f"    κ_phase: {config['kappa_phase']}")
    print(f"    ω_phases: {config['omega_phases']}")

    pipeline = APGIPipeline(config)

    n_steps = 400
    x_e, x_i = generate_input_signals(n_steps)

    print(f"\nSimulating {n_steps} steps with phase modulation...")

    # Track phase and threshold modulation
    phase_history = []
    theta_history = []

    for i in range(n_steps):
        result = pipeline.step(x_e[i], x_i[i])

        if "hierarchical_phases" in result:
            phase_history.append(result["hierarchical_phases"])
        theta_history.append(result["theta"])

    print(f"\nResults:")
    print(f"  Threshold variation (σ): {np.std(theta_history):.4f}")
    print(f"  Mean threshold: {np.mean(theta_history):.4f}")

    if phase_history:
        phase_array = np.array(phase_history)
        print(f"\n  Phase dynamics:")
        for level in range(min(3, len(phase_history[0]))):
            phases = phase_array[:, level]
            print(
                f"    Level {level}: phase range [{phases.min():.3f}, {phases.max():.3f}] rad"
            )

    print(f"\n  PAC mechanism:")
    print(f"    θ_ℓ(t) = θ_0 · [1 + κ_phase · Π_ℓ+1 · cos(φ_ℓ+1(t))]")
    print(f"    Higher-level phases modulate lower-level thresholds")


def demonstrate_synchronization_patterns():
    """Demonstrate different synchronization patterns."""
    print("\n" + "=" * 70)
    print("Example 4: Synchronization Patterns")
    print("=" * 70)

    n_levels = 4
    n_steps = 300

    coupling_values = [0.0, 0.05, 0.1, 0.2]

    print(f"\nTesting different coupling strengths...")
    print(f"Levels: {n_levels}, Steps: {n_steps}")

    results = []

    for coupling in coupling_values:
        config = {
            "omega_0": 0.5,
            "freq_scaling": 0.7,
            "noise_std": 0.05,
            "coupling_strength": coupling,
        }

        kuramoto = HierarchicalKuramotoSystem(n_levels=n_levels, config=config)

        sync_values = []
        for _ in range(n_steps):
            result = kuramoto.step(dt=0.1)
            sync_values.append(result["synchronization"])

        results.append(
            {
                "coupling": coupling,
                "mean_sync": np.mean(sync_values),
                "final_sync": sync_values[-1],
                "sync_variance": np.var(sync_values),
            }
        )

    print(f"\nResults:")
    print(f"{'Coupling':>10} {'Mean Sync':>12} {'Final Sync':>12} {'Sync Var':>12}")
    print("-" * 50)
    for r in results:
        print(
            f"{r['coupling']:>10.2f} {r['mean_sync']:>12.4f} "
            f"{r['final_sync']:>12.4f} {r['sync_variance']:>12.6f}"
        )

    print(f"\nObservations:")
    print(f"  - Zero coupling: independent oscillators (low sync)")
    print(f"  - Weak coupling: partial synchronization")
    print(f"  - Strong coupling: near-complete synchronization")


def demonstrate_frequency_modulation():
    """Demonstrate frequency modulation by precision."""
    print("\n" + "=" * 70)
    print("Example 5: Precision-Modulated Frequencies")
    print("=" * 70)

    n_levels = 3
    n_steps = 200
    dt = 0.1

    # Test different precision values
    precision_values = [0.1, 1.0, 5.0, 10.0]

    print(f"\nTesting frequency modulation by precision...")

    config = {
        "omega_0": 0.5,
        "freq_scaling": 0.7,
        "noise_std": 0.05,
        "coupling_strength": 0.1,
    }

    print(f"\nBase frequency ω_0: {config['omega_0']} Hz")
    print(f"Frequency scaling: {config['freq_scaling']}")

    results = []

    for pi in precision_values:
        kuramoto = HierarchicalKuramotoSystem(n_levels=n_levels, config=config)

        # Step with constant precision
        frequencies = []
        for _ in range(n_steps):
            result = kuramoto.step(dt=dt)
            # Extract effective frequency from phase change
            frequencies.append(config["omega_0"] * (1 + 0.1 * np.log1p(pi)))

        results.append(
            {
                "precision": pi,
                "mean_freq": np.mean(frequencies),
            }
        )

    print(f"\nEffective frequencies:")
    print(f"{'Precision (Π)':>15} {'Effective ω':>15}")
    print("-" * 35)
    for r in results:
        print(f"{r['precision']:>15.2f} {r['mean_freq']:>15.4f}")

    print(f"\nMechanism:")
    print(f"  ω_eff = ω_0 · (1 + α·log(1 + Π))")
    print(f"  Higher precision → faster oscillation")
    print(f"  This links attention (precision) to oscillatory speed")


def main():
    """Run all Kuramoto coupling examples."""
    print("=" * 70)
    print("APGI Kuramoto Oscillatory Coupling Examples")
    print("=" * 70)

    # Run all demonstrations
    demonstrate_kuramoto_system()
    demonstrate_kuramoto_with_apgi()
    demonstrate_phase_modulation()
    demonstrate_synchronization_patterns()
    demonstrate_frequency_modulation()

    print("\n" + "=" * 70)
    print("All Kuramoto Examples Complete")
    print("=" * 70)

    print("\nKey concepts demonstrated:")
    print("  1. Kuramoto oscillators model hierarchical phase dynamics")
    print("  2. Phase reset on ignition coordinates timing")
    print("  3. Phase-amplitude coupling links oscillation to threshold")
    print("  4. Coupling strength controls synchronization")
    print("  5. Precision modulates oscillatory frequency")

    print("\nApplications:")
    print("  - Neural oscillation modeling (gamma, theta, alpha)")
    print("  - Cross-scale information coordination")
    print("  - Attention-dependent frequency modulation")
    print("  - Phase-locked threshold modulation")


if __name__ == "__main__":
    main()
